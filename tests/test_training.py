"""Validate the trainer's self-test, the HRCrackNet network and loss, and the server additions
(tolerance ring, crash-safe checkpoints, resumed epoch logs, saved overlays, initial weights)."""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import numpy as np
    import torch
    import segmentation_models_pytorch as smp
    from hrcracknet.evaluation import (
        ThresholdSweep, counts_at_indices, metrics_from_counts, prediction_writer, source_threshold_indices,
        spread_over_sources,
    )
    from hrcracknet.losses import FocalCrackLoss, HairlineCrackLoss, distance_loss
    from hrcracknet.models import HRCrackNet, StripContextBlock, encoder_arguments, load_model_from_checkpoint
    from hrcracknet.self_test import run_self_tests
    from hrcracknet.training import initial_weights, open_epoch_log, parameter_groups, save_checkpoint
except ImportError:  # the workflow tests still run without PyTorch and segmentation_models_pytorch
    torch = None


@unittest.skipIf(torch is None, 'PyTorch or segmentation_models_pytorch is not installed')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.target = torch.zeros(1, 1, 32, 32)
        self.target[0, 0, 16, 4:28] = 1  # a one-pixel horizontal crack on row 16
        self.valid = torch.ones_like(self.target)

    def gradient(self, tolerance, loss=None):
        logits = torch.randn(1, 1, 32, 32, requires_grad=True)
        (loss or HairlineCrackLoss)(label_tolerance=tolerance)(logits, self.target, self.valid).backward()
        return logits.grad[0, 0]

    def test_built_in_self_test_passes(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_self_tests()  # raises SystemExit with the failed checks
        self.assertIn('All self-tests passed.', output.getvalue())

    def test_strict_loss_supervises_pixels_beside_the_label(self):
        self.assertTrue((self.gradient(0)[14:19, 4:28] != 0).all())

    def test_tolerance_ring_is_left_out_of_the_loss(self):
        for loss in (HairlineCrackLoss, FocalCrackLoss):
            gradient = self.gradient(2, loss)
            self.assertTrue((gradient[[14, 15, 17, 18], 4:28] == 0).all())  # within 2 px of the label
            self.assertTrue((gradient[16, 4:28] != 0).all())                # the label itself
            self.assertTrue((gradient[[13, 19], 4:28] != 0).all())          # 3 px away: background again

    def test_checkpoint_is_replaced_whole(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            save_checkpoint({'epoch': 1}, path)
            save_checkpoint({'epoch': 2}, path)
            self.assertEqual(torch.load(path, weights_only=False)['epoch'], 2)
            self.assertEqual([item.name for item in Path(folder).iterdir()], ['last.pt'])

    def test_resumed_log_drops_the_repeated_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'training_history.csv'
            # Killed after logging epoch 3 but before saving last.pt, so epoch 3 runs again.
            path.write_text('epoch,loss\n1,0.9\n2,0.8\n3,0.7\n', encoding='utf-8')
            with open_epoch_log(path, ['epoch', 'loss'], 3) as handle:
                handle.write('3,0.6\n')
            self.assertEqual(path.read_text(encoding='utf-8').split(), ['epoch,loss', '1,0.9', '2,0.8', '3,0.6'])
            with open_epoch_log(path, ['epoch', 'loss'], 1):  # a fresh run starts the file again
                pass
            self.assertEqual(path.read_text(encoding='utf-8').split(), ['epoch,loss'])

    def test_saved_overlays_cover_every_source(self):
        domain_ids = [0] * 300 + [1] * 50 + [2] * 3  # one large source first, as in the size-grouped loader
        chosen = spread_over_sources(domain_ids, 40)
        per_source = [sum(domain_ids[index] == domain for index in chosen) for domain in range(3)]
        self.assertEqual(per_source, [19, 18, 3])  # the small source is saved whole, the rest shared
        self.assertGreater(max(index for index in chosen if domain_ids[index] == 0), 250)  # not just the first files
        self.assertEqual(spread_over_sources(domain_ids, 10**6), set(range(len(domain_ids))))  # a large limit saves all

    def test_every_source_gets_its_own_threshold(self):
        # Source 0 is predicted confidently, source 1 (faint cracks) only weakly, source 2 has few images.
        thresholds = [0.05, 0.25, 0.5, 0.75, 0.95]
        sweep = ThresholdSweep(thresholds, 0, 3, 'cpu')
        target = torch.zeros(1, 1, 8, 8)
        target[..., 2:4, :] = 1
        strong, weak = target * 0.6 + 0.3, target * 0.1 + 0.02  # crack 0.9 on background 0.3; 0.12 on 0.02
        for probability, domain in ((strong, 0), (weak, 1), (weak, 2)):
            sweep.update(probability, target, torch.ones_like(target), torch.tensor([domain]))
        images = {0: 40, 1: 40, 2: 3}
        chosen = source_threshold_indices(sweep, [0, 1, 2], 'f1', images, 10, fallback=2)
        self.assertEqual(chosen, {0: 2, 1: 0, 2: 2})  # above its background; the lowest; three images: shared
        combined = counts_at_indices(sweep, chosen)
        expected = sum(sweep.count_array([domain])[index] for domain, index in chosen.items())
        self.assertTrue(np.array_equal(combined[0], expected))
        metrics = metrics_from_counts(combined)
        self.assertAlmostEqual(float(metrics['recall'][0]), 2 / 3, places=4)  # the small source is missed at 0.5
        self.assertAlmostEqual(float(metrics['precision'][0]), 1.0, places=4)

    def test_overlays_are_written_for_the_selected_images_only(self):
        class Tiles:  # the parts of CrackSegmentationDataset the writer reads
            domain_ids = [0, 0, 0, 1]
            shapes = [(32, 32)] * 4
            pairs = [(Path(f'source{domain}_{index}.png'), None) for index, domain in enumerate(domain_ids)]

        with tempfile.TemporaryDirectory() as folder:
            write = prediction_writer(Tiles, Path(folder), 2, 0.5)
            write(torch.rand(4, 3, 32, 32), torch.rand(4, 1, 32, 32), self.target.expand(4, -1, -1, -1),
                  torch.arange(4))
            names = sorted(item.name for item in Path(folder).iterdir())
            self.assertEqual(names, ['source0_0_overlay.png', 'source0_0_probability.png',
                                     'source1_3_overlay.png', 'source1_3_probability.png'])


@unittest.skipIf(torch is None, 'PyTorch or segmentation_models_pytorch is not installed')
class HRCrackNetTests(unittest.TestCase):
    # The smallest encoder of each family; the arguments are those the trainer stores in a checkpoint.
    hrnet, convnext = 'tu-hrnet_w18_small', 'tu-convnext_atto'
    kwargs = dict(encoder_name=hrnet, context_channels=128, strip_blocks=2, strip_length=9, pool_length=31,
                  decoder_channels=[64, 32], half_channels=48, stem_channels=32, line_prior=True,
                  deep_supervision=True, distance_head=True, drop_path_rate=0.1)

    @staticmethod
    def build(**changes):
        torch.manual_seed(0)
        return HRCrackNet(pretrained=False, **{**HRCrackNetTests.kwargs, **changes})

    def test_hrnet_gives_its_branch_outputs(self):
        self.assertEqual(encoder_arguments('tu-hrnet_w32', 0.1), {'feature_location': ''})  # no stochastic depth
        self.assertEqual(encoder_arguments('tu-convnext_tiny', 0.1), {'drop_path_rate': 0.1})
        self.assertEqual(encoder_arguments('efficientnet-b4', 0.1), {})
        model = self.build()
        self.assertEqual(list(model.encoder.out_channels), [3, 64, 16, 32, 64, 128])
        self.assertIsNone(model.half_resolution)  # HRNet has 1/2-scale features of its own
        self.assertEqual(model.aggregate[0].in_channels, 16 + 32 + 64 + 128)

    def test_convnext_gets_the_half_scale_features_it_lacks(self):
        model = self.build(encoder_name=self.convnext).eval()
        self.assertEqual(list(model.encoder.out_channels)[1], 0)   # smp's mark of the missing 1/2 scale
        self.assertIsNotNone(model.half_resolution)
        self.assertEqual(model.up_half[0][0].in_channels, 128 + 48)  # the stem's channels are the 1/2-scale skip
        with torch.no_grad():
            self.assertEqual(model(torch.rand(1, 3, 64, 96)).shape, (1, 1, 64, 96))

    def test_outputs_in_training_and_evaluation(self):
        images = torch.rand(2, 3, 64, 96)
        for encoder in (self.hrnet, self.convnext):
            logits, auxiliary, distance = self.build(encoder_name=encoder).train()(images)
            self.assertEqual([output.shape for output in (logits, *auxiliary, distance)], [(2, 1, 64, 96)] * 4)
        logits, auxiliary, distance = self.build(deep_supervision=False, distance_head=False).train()(images)
        self.assertEqual((auxiliary, distance), ([], None))
        with torch.no_grad():
            self.assertEqual(self.build().eval()(images).shape, (2, 1, 64, 96))

    def test_auxiliary_weights_and_distance_weight_steer_the_heads(self):
        model = self.build().train()
        images, masks = torch.rand(2, 3, 64, 96), (torch.rand(2, 1, 64, 96) > 0.9).float()
        loss = HairlineCrackLoss(auxiliary_weights=(0.5, 0.0), distance_weight=0.0)
        loss(model(images), masks, torch.ones_like(masks)).backward()
        first, second = (head.weight.grad.abs().sum().item() for head in model.auxiliary_heads)
        self.assertGreater(first, 0.0)
        self.assertEqual(second, 0.0)  # a zero weight leaves that output untrained
        self.assertIsNone(model.distance_head.weight.grad)  # weight 0: the head is not in the loss at all
        model.zero_grad()
        HairlineCrackLoss()(model(images), masks, torch.ones_like(masks)).backward()
        self.assertGreater(model.distance_head.weight.grad.abs().sum().item(), 0.0)

    def test_the_parts_can_be_left_out(self):
        full, baseline = self.build(), self.build(strip_blocks=0, stem_channels=0, deep_supervision=False,
                                                  distance_head=False)
        self.assertEqual(len(full.context), 2)
        self.assertEqual(len(baseline.context), 0)
        self.assertEqual(full.up_full[0][0].in_channels, 64 + 32)
        self.assertEqual(baseline.up_full[0][0].in_channels, 64)
        self.assertIsNone(baseline.full_resolution)
        self.assertIsNone(self.build(line_prior=False).thin_lines)
        self.assertEqual(self.build(line_prior=False).full_resolution[0][0].in_channels, 3)
        self.assertEqual(full.full_resolution[0][0].in_channels, 7)  # RGB and four thin-line responses
        self.assertIsNone(baseline.auxiliary_heads)
        self.assertIsNone(baseline.distance_head)
        self.assertLess(sum(p.numel() for p in baseline.parameters()), sum(p.numel() for p in full.parameters()))

    def test_strip_context_does_not_depend_on_the_tile_size(self):
        # The pooling segments have a fixed length, so a pixel far from the border gets the same
        # value in a 256-px training crop and in a larger evaluation tile.
        block = StripContextBlock(8, 9, 31).eval()
        torch.manual_seed(1)
        large = torch.rand(1, 8, 192, 192)
        with torch.no_grad():
            whole, crop = block(large), block(large[..., 64:128, 64:128].contiguous())
        self.assertTrue(torch.allclose(whole[..., 64 + 24:64 + 40, 64 + 24:64 + 40], crop[..., 24:40, 24:40], atol=1e-5))

    def test_distance_loss_rewards_a_prediction_near_the_crack(self):
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, 16, 4:28] = 1
        valid = torch.ones_like(target)
        on_the_crack = torch.where(target > 0, 4.0, -4.0)
        one_pixel_off = torch.roll(on_the_crack, 1, dims=2)
        far_away = torch.roll(on_the_crack, 10, dims=2)
        losses = [float(distance_loss(logits, target, valid, 5)) for logits in (on_the_crack, one_pixel_off, far_away)]
        self.assertLess(losses[1], losses[2])  # a miss by one pixel costs less than a miss by ten
        self.assertLess(losses[0], losses[2])

    def test_only_the_encoder_trains_at_the_reduced_rate(self):
        model = self.build()
        groups = parameter_groups(model, 3e-4, 0.1, 1e-4)
        rate = {id(parameter): group['lr'] for group in groups for parameter in group['params']}
        for name, parameter in model.named_parameters():
            self.assertAlmostEqual(rate[id(parameter)], 3e-5 if name.startswith('encoder.') else 3e-4, msg=name)
        self.assertTrue(any(name.startswith('context.') for name, _ in model.named_parameters()))

    def test_checkpoint_rebuilds_the_network(self):
        for changes in ({}, {'encoder_name': self.convnext, 'strip_blocks': 0, 'stem_channels': 0, 'line_prior': False}):
            model = self.build(**changes).eval()
            architecture = {'name': 'hrcracknet', 'kwargs': {**self.kwargs, **changes}}
            images = torch.rand(1, 3, 64, 64)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'best.pt'
                save_checkpoint({'model': model.state_dict(), 'architecture': architecture}, path)
                rebuilt, _ = load_model_from_checkpoint(path)
                with torch.no_grad():
                    self.assertTrue(torch.equal(rebuilt(images), model(images)))

    def test_initial_weights_need_the_same_network(self):
        trained = self.build().eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            save_checkpoint({'model': trained.state_dict(),
                             'architecture': {'name': 'hrcracknet', 'kwargs': self.kwargs}}, path)
            # Tuples for lists and another drop-path rate are the same network.
            wanted = {**self.kwargs, 'decoder_channels': (64, 32), 'drop_path_rate': 0.0}
            model = HRCrackNet(pretrained=False, **wanted).eval()
            keys = model.load_state_dict(initial_weights(path, wanted)['model'], strict=False)
            self.assertEqual((keys.missing_keys, keys.unexpected_keys), ([], []))
            images = torch.rand(2, 3, 64, 96)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(images), trained(images)))
            for change in ({'encoder_name': self.convnext}, {'strip_blocks': 0}, {'stem_channels': 0},
                           {'deep_supervision': False}, {'distance_head': False}, {'decoder_channels': [32, 16]},
                           {'pool_length': 15}):
                with self.assertRaises(SystemExit):
                    initial_weights(path, {**self.kwargs, **change})


if __name__ == '__main__':
    unittest.main()
