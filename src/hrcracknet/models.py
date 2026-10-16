'Models for hairline crack segmentation.'

from __future__ import annotations
from .common import DEFAULT_ENCODER, F, copy, math, nn, torch

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class ConvBNReLU(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__(ConvBNReLU(input_channels, output_channels, stride=stride),
                         ConvBNReLU(output_channels, output_channels))


class ThinLineResponse(nn.Module):
    """Fixed morphological black-hat and white top-hat on grayscale input.

    Black-hat (closing - image) responds to dark structures narrower than the
    kernel; top-hat (image - opening) to bright ones. Both are needed: in EH104
    many cracks appear lighter than the concrete (deposits in the crack), not
    darker. The module has no parameters, so it is a thin-line prior at no data cost.
    """

    def __init__(self, kernel_sizes=(7, 15)):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.output_channels = 2 * len(self.kernel_sizes)
        self.register_buffer("luma", torch.tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1))

    def forward(self, rgb):
        gray = (rgb * self.luma).sum(1, keepdim=True)
        responses = []
        for size in self.kernel_sizes:
            closed = -F.max_pool2d(-F.max_pool2d(gray, size, 1, size // 2), size, 1, size // 2)
            opened = F.max_pool2d(-F.max_pool2d(-gray, size, 1, size // 2), size, 1, size // 2)
            responses.extend((closed - gray, gray - opened))
        return torch.cat(responses, 1)


class DirectionalStripConv(nn.Module):
    """Depthwise convolutions along four straight lines: horizontal, vertical and both diagonals.

    The strip convolution of CoANet (Mei et al., 2021), made for roads in aerial images: each of
    the four outputs of a channel is a weighted sum of `length` pixels on one line through the
    pixel. A hairline crack is a long run of small, consistent dips; a sum along its direction
    adds the dips up while surface texture, which does not line up, averages out. A square kernel
    of the same reach would mix in the texture beside the crack.

    The diagonal kernels are built from `length` weights each (``diag_embed``), so all four
    directions have the same number of parameters. Output: the four directions concatenated,
    (N, 4 C, H, W), in the order of DIRECTIONS.
    """

    DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti-diagonal")

    def __init__(self, channels: int, length: int = 9):
        super().__init__()
        if length < 3 or length % 2 == 0:
            raise ValueError(f"The strip length must be odd and at least 3, not {length}.")
        self.channels, self.length = channels, length
        self.weight = nn.Parameter(torch.empty(len(self.DIRECTIONS), channels, length))
        nn.init.normal_(self.weight, 0.0, math.sqrt(2.0 / length))  # He initialisation, fan-in `length`

    def kernels(self) -> list:
        """The four depthwise kernels: (C, 1, 1, L), (C, 1, L, 1), and (C, 1, L, L) twice."""
        horizontal, vertical, diagonal, anti = self.weight.unsqueeze(2)  # each (C, 1, L)
        diagonal = torch.diag_embed(diagonal)                            # (C, 1, L, L)
        return [horizontal.unsqueeze(2), vertical.unsqueeze(3), diagonal, torch.flip(torch.diag_embed(anti), (-1,))]

    def forward(self, x):
        half = self.length // 2
        paddings = ((0, half), (half, 0), (half, half), (half, half))
        return torch.cat([F.conv2d(x, kernel.to(x.dtype), padding=padding, groups=self.channels)
                          for kernel, padding in zip(self.kernels(), paddings)], 1)


class StripContextBlock(nn.Module):
    """Residual block of directional strip convolutions, gated by local strip pooling.

    ``x -> 1x1 (C -> C/2) -> four strips (C/2 each) -> 1x1 (2C -> C) -> times gate -> + x``.

    The gate is the strip pooling of SPNet (Hou et al., 2020): the mean over a row segment and
    over a column segment of `pool_length` pixels, a 3-pixel convolution across each, their sum
    and a sigmoid. It weighs a response by how it continues along the row and the column: a
    formwork joint or a drip stain runs straight across the tile and fills its whole row segment,
    while a crack wanders. SPNet pools whole rows; here the segment has a fixed length, because
    the network is trained on 256-px crops and evaluated on tiles of up to 1024 px, and a mean
    over the whole row would change with the size of the input.
    """

    def __init__(self, channels: int, length: int = 9, pool_length: int = 31):
        super().__init__()
        if channels < 2 or pool_length < 1 or pool_length % 2 == 0:
            raise ValueError(f"Need at least 2 channels and an odd pool length, not {channels} and {pool_length}.")
        hidden = channels // 2
        self.pool_length = pool_length
        self.reduce = ConvBNReLU(channels, hidden, 1)
        self.strips = DirectionalStripConv(hidden, length)
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(4 * hidden), nn.ReLU(inplace=True),
            nn.Conv2d(4 * hidden, channels, 1, bias=False), nn.BatchNorm2d(channels),
        )
        self.across_rows = nn.Sequential(nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), bias=False),
                                         nn.BatchNorm2d(channels))
        self.across_columns = nn.Sequential(nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), bias=False),
                                            nn.BatchNorm2d(channels))
        self.gate = nn.Conv2d(channels, channels, 1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        response = self.fuse(self.strips(self.reduce(x)))
        half = self.pool_length // 2
        rows = F.avg_pool2d(x, (1, self.pool_length), 1, (0, half), count_include_pad=False)
        columns = F.avg_pool2d(x, (self.pool_length, 1), 1, (half, 0), count_include_pad=False)
        gate = torch.sigmoid(self.gate(F.relu(self.across_rows(rows) + self.across_columns(columns))))
        return self.activation(x + response * gate)


def smp_library():
    """segmentation_models_pytorch, imported when a network is built."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError("The network requires segmentation_models_pytorch: "
                           "python -m pip install segmentation-models-pytorch==0.5.0 timm") from exc
    return smp


def encoder_arguments(encoder_name: str, drop_path_rate: float = 0.0) -> dict:
    """The timm arguments smp passes on when it builds `encoder_name`.

    HRNet (``tu-hrnet_*``) gets ``feature_location=""``: the outputs of its four parallel branches
    (32, 64, 128 and 256 channels at 1/4 to 1/32 scale for W32), which HRNetV2 segments with.
    timm's default for HRNet passes each branch through a classification bottleneck first (128
    to 1024 channels), which adds 1.6 M parameters that segmentation does not use. HRNet has no
    stochastic depth; `drop_path_rate` is passed to the other timm encoders (ConvNeXt).
    """
    if encoder_name.startswith("tu-hrnet"):
        return {"feature_location": ""}
    if encoder_name.startswith("tu-") and drop_path_rate > 0:
        return {"drop_path_rate": drop_path_rate}
    return {}


def encoder_statistics(encoder_name: str) -> tuple[tuple, tuple]:
    """Mean and standard deviation of RGB in [0, 1] that the pretrained encoder expects.

    Most encoders were trained with the ImageNet statistics, but not all: the ``tf_`` ports of
    EfficientNet in timm use 0.5 for both. timm encoders (``tu-<name>``) carry the values in
    their pretrained configuration and smp's own encoders in smp's settings table.
    """
    smp = smp_library()
    if encoder_name.startswith("tu-"):
        import timm

        configuration = timm.models.get_pretrained_cfg(encoder_name[3:])
        if configuration is not None:
            return tuple(configuration.mean), tuple(configuration.std)
        return IMAGENET_MEAN, IMAGENET_STD
    settings = getattr(smp.encoders, "pretrained_settings", {}).get(encoder_name, {}).get("imagenet", {})
    return tuple(settings.get("mean", IMAGENET_MEAN)), tuple(settings.get("std", IMAGENET_STD))


def upsample(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class HRCrackNet(nn.Module):
    """High-resolution crack network: HRNet, directional strip context and a distance-field head.

    Built for cracks one or two pixels wide whose contrast is close to that of the surface texture,
    from four parts, each with its own switch for the ablation study:

    * Encoder (`encoder_name`, default ``tu-hrnet_w32``): HRNet-W32 (Wang et al., 2020),
      ImageNet-pretrained, through segmentation_models_pytorch (smp). A U-Net encoder halves the
      resolution stage after stage and a decoder has to recover thin cracks from coarse features;
      HRNet keeps a 1/4-scale branch through the whole network and exchanges information between
      its four branches (1/4 to 1/32) at every stage. HrSegNet (Li et al., 2023) uses the same
      principle for cracks. Any other smp encoder works with the same decoder, which the study
      uses to measure the encoder's share (``tu-convnext_tiny``, the UNet++ encoder). An encoder
      without 1/2-scale features (ConvNeXt) gets two 3x3 convolutions, the first with stride 2
      (`half_channels`), as in the UNet++ project.
    * Aggregation (HRNetV2): every encoder output from 1/4 to 1/32 scale brought to 1/4 scale,
      concatenated and reduced to `context_channels` by a 1x1 convolution.
    * Strip context (`strip_blocks` StripContextBlocks, 0 to switch it off): directional strip
      convolutions of `strip_length` pixels at 1/4 scale (CoANet), gated by local strip pooling
      over `pool_length` pixels (SPNet).
    * Decoder: up to 1/2 scale with the encoder's 1/2-scale feature as skip, two 3x3 convolutions
      (`decoder_channels[0]`); up to full resolution with the hairline stem of the other trainers
      as skip (`stem_channels`, fed with fixed black-hat and top-hat responses if `line_prior`),
      two 3x3 convolutions (`decoder_channels[1]`); a 3x3 output convolution.

    Outputs used in training only:

    * Deep supervision (`deep_supervision`): a 1x1 crack output on the 1/4-scale context and on
      the 1/2-scale decoder features.
    * Distance field (`distance_head`): a 3x3 convolution on the full-resolution features
      regresses how close each pixel is to a crack (``losses.distance_field``). A crack label one
      or two pixels wide gives few positive pixels, and a label drawn a pixel off turns them all
      into misses; the distance field changes gradually around the crack and gives the network a
      target in its neighbourhood too (deep distance transform, Wang et al., 2020).

    Input: RGB in [0, 1], height and width multiples of 32; the image is normalised inside with the
    statistics the encoder was pretrained with. Eval mode returns the crack logits (N, 1, H, W).
    Train mode returns ``(logits, auxiliary, distance)``: the full-size logits of the deep-supervision
    outputs (1/4 then 1/2 scale; empty without deep supervision) and the full-size distance logits
    (None without the distance head).
    """

    def __init__(
        self, encoder_name: str = DEFAULT_ENCODER, pretrained: bool = True, context_channels: int = 128,
        strip_blocks: int = 2, strip_length: int = 9, pool_length: int = 31, decoder_channels=(64, 32),
        half_channels: int = 48, stem_channels: int = 32, line_prior: bool = True, deep_supervision: bool = True,
        distance_head: bool = True, drop_path_rate: float = 0.0, prior_probability: float = 0.03,
    ):
        super().__init__()
        smp = smp_library()
        from segmentation_models_pytorch.base import initialization

        decoder_channels = tuple(int(channels) for channels in decoder_channels)
        if len(decoder_channels) != 2 or min(decoder_channels) < 1 or context_channels < 2:
            raise ValueError(f"decoder_channels needs two positive values (1/2 and 1/1 scale), not "
                             f"{decoder_channels}, and context_channels at least 2, not {context_channels}.")
        try:
            self.encoder = smp.encoders.get_encoder(
                encoder_name, in_channels=3, depth=5, weights="imagenet" if pretrained else None,
                **encoder_arguments(encoder_name, drop_path_rate))
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not build the {encoder_name} encoder with its pretrained weights ({exc}). Compute "
                    f"nodes cannot download: cache them on the login node with "
                    f"'ENCODER={encoder_name} bash scripts/setup_conda.sh', or pass --no-pretrained."
                ) from exc
            raise
        channels = list(self.encoder.out_channels)  # image, then the features at 1/2, 1/4, 1/8, 1/16, 1/32 scale
        if len(channels) != 6 or min(channels[2:]) < 1:
            raise ValueError(f"{encoder_name} does not give features at 1/4 to 1/32 scale (channels {channels}).")
        self.half_resolution = None
        if channels[1] == 0:  # ConvNeXt: no 1/2-scale features
            self.half_resolution = DoubleConv(3, half_channels, stride=2)
            channels[1] = half_channels
        half, full = decoder_channels
        self.aggregate = ConvBNReLU(sum(channels[2:]), context_channels, 1)
        self.context = nn.Sequential(*(StripContextBlock(context_channels, strip_length, pool_length)
                                       for _ in range(strip_blocks)))
        self.up_half = DoubleConv(context_channels + channels[1], half)
        self.thin_lines, self.full_resolution = None, None
        skip = 0
        if stem_channels > 0:
            self.thin_lines = ThinLineResponse() if line_prior else None
            self.full_resolution = DoubleConv(3 + (self.thin_lines.output_channels if line_prior else 0), stem_channels)
            skip = stem_channels
        self.up_full = DoubleConv(half + skip, full)
        for module in (self.half_resolution, self.aggregate, self.context, self.up_half, self.full_resolution,
                       self.up_full):
            if module is not None:
                initialization.initialize_decoder(module)  # He-uniform convolutions, BatchNorm 1 and 0
        self.head = nn.Conv2d(full, 1, 3, padding=1)
        self.auxiliary_heads = nn.ModuleList([nn.Conv2d(context_channels, 1, 1), nn.Conv2d(half, 1, 1)]
                                             ) if deep_supervision else None
        self.distance_head = nn.Conv2d(full, 1, 3, padding=1) if distance_head else None
        # Start near the crack prior so early training is not dominated by background. Nearly every
        # pixel is far from a crack, where the distance field is 0, so its output starts there too.
        bias = -math.log((1 - prior_probability) / prior_probability)
        for head in [self.head, *(self.auxiliary_heads or []), *([self.distance_head] if distance_head else [])]:
            initialization.initialize_head(head)
            nn.init.constant_(head.bias, bias)
        mean, std = encoder_statistics(encoder_name)
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1))

    def features(self, x) -> dict:
        """The decoder features: ``quarter`` (the strip context, 1/4 scale), ``half`` and ``full``."""
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError(f"Height and width must be multiples of 32, not {tuple(x.shape[-2:])}.")
        normalized = (x - self.image_mean) / self.image_std
        features = list(self.encoder(normalized))
        if self.half_resolution is not None:
            features[1] = self.half_resolution(normalized)
        quarter = features[2].shape[-2:]
        joined = torch.cat([features[2], *(upsample(feature, quarter) for feature in features[3:])], 1)
        context = self.context(self.aggregate(joined))
        half = self.up_half(torch.cat((upsample(context, features[1].shape[-2:]), features[1]), 1))
        full = upsample(half, x.shape[-2:])
        if self.full_resolution is not None:
            stem_input = normalized if self.thin_lines is None else torch.cat((normalized, self.thin_lines(x)), 1)
            full = torch.cat((full, self.full_resolution(stem_input)), 1)
        return {"quarter": context, "half": half, "full": self.up_full(full)}

    def forward(self, x):
        features = self.features(x)
        logits = self.head(features["full"])
        if not self.training:
            return logits
        size = x.shape[-2:]
        auxiliary = [] if self.auxiliary_heads is None else [
            upsample(head(features[name]), size) for head, name in zip(self.auxiliary_heads, ("quarter", "half"))
        ]
        distance = None if self.distance_head is None else self.distance_head(features["full"])
        return logits, auxiliary, distance


# Checkpoint "architecture" names and the networks they rebuild.
ARCHITECTURES = {"hrcracknet": HRCrackNet}
# The arguments that decide the layers of the network: two runs share weights only if these agree.
NETWORK_ARGUMENTS = ("encoder_name", "context_channels", "strip_blocks", "strip_length", "pool_length",
                     "decoder_channels", "half_channels", "stem_channels", "line_prior", "deep_supervision",
                     "distance_head")


def different_arguments(saved: dict, wanted: dict) -> dict:
    """The NETWORK_ARGUMENTS in which a checkpoint's network differs from the one wanted, with the saved values."""
    return {key: saved[key] for key in NETWORK_ARGUMENTS
            if key in saved and key in wanted and json_value(saved[key]) != json_value(wanted[key])}


def json_value(value):
    """Lists and tuples compare equal: a checkpoint read back holds what was saved as either."""
    return list(value) if isinstance(value, (list, tuple)) else value


def load_model_from_checkpoint(path, device="cpu"):
    """Rebuild the network of a best.pt for inference scripts.

    The architecture kwargs in the checkpoint carry the encoder and the decoder options. The
    model is called as ``model(images)`` with RGB in [0, 1] and returns the crack logits.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint["architecture"]
    model = ARCHITECTURES[architecture["name"]](pretrained=False, **architecture["kwargs"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


class ModelEMA:
    """Exponential moving average of weights (and BN statistics)."""

    def __init__(self, model: nn.Module, decay: float = 0.999, ramp_steps: int = 2000):
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay, self.ramp_steps, self.updates = decay, ramp_steps, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.ramp_steps))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


__all__ = [
    'IMAGENET_MEAN', 'IMAGENET_STD', 'ConvBNReLU', 'DoubleConv', 'ThinLineResponse', 'DirectionalStripConv',
    'StripContextBlock', 'smp_library', 'encoder_arguments', 'encoder_statistics', 'upsample', 'HRCrackNet',
    'ARCHITECTURES', 'NETWORK_ARGUMENTS', 'different_arguments', 'json_value', 'load_model_from_checkpoint',
    'ModelEMA',
]
