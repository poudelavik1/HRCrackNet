'Losses for hairline crack segmentation.'

from __future__ import annotations
from .common import F, nn, torch

# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def soft_erode(image):
    return torch.minimum(
        -F.max_pool2d(-image, (3, 1), 1, (1, 0)), -F.max_pool2d(-image, (1, 3), 1, (0, 1))
    )


def soft_skeleton(image, iterations: int):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021)."""
    opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(probability, target, iterations: int, smooth: float = 1.0):
    predicted_skeleton = soft_skeleton(probability, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    topology_precision = ((predicted_skeleton * target).sum() + smooth) / (predicted_skeleton.sum() + smooth)
    topology_sensitivity = ((target_skeleton * probability).sum() + smooth) / (target_skeleton.sum() + smooth)
    return 1 - 2 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity)


def distance_field(target, margin: int):
    """Closeness to the nearest crack pixel: 1 on the crack, falling by 1 / (margin + 1) per pixel
    of chessboard distance, and 0 from margin + 1 pixels on.

    Built with repeated 3x3 max pooling on the GPU, inside the loss, so the data pipeline (shared
    with the other trainers) stays as it is and the field follows every augmentation of the mask.
    """
    field, grown = target, target
    for _ in range(margin):
        grown = F.max_pool2d(grown, 3, 1, 1)
        field = field + grown
    return field / (margin + 1)


def distance_loss(logits, target, valid, margin: int):
    """Balanced L1 between the predicted closeness (sigmoid of `logits`) and distance_field(target).

    The pixels near a crack (field > 0) and the others are averaged separately and weighted
    equally, so the few pixels around a hairline count as much as the wall.
    """
    field = distance_field(target, margin)
    error = (torch.sigmoid(logits.float()) - field).abs() * valid
    near = (field > 0).float() * valid
    far = valid - near
    return 0.5 * ((error * near).sum() / near.sum().clamp_min(1.0) + (error * far).sum() / far.sum().clamp_min(1.0))


# The terms --loss combines. bce, focal, dice and tversky are region terms (every pixel counts);
# cldice follows the crack centreline and is only meaningful next to a region term: on its own its
# minimum is a prediction of crack everywhere, whose soft skeleton is empty.
LOSS_TERMS = ("bce", "focal", "dice", "tversky", "cldice")
REGION_TERMS = ("bce", "focal", "dice", "tversky")
# A name for several terms: "hairline" is the loss of the other hairline trainers.
LOSS_ALIASES = {"hairline": ("bce", "tversky", "cldice")}
TERM_LABELS = {"bce": "BCE", "focal": "Focal", "dice": "Dice", "tversky": "Tversky", "cldice": "clDice"}
DEFAULT_TERM_WEIGHTS = {"bce": 1.0, "focal": 1.0, "dice": 1.0, "tversky": 1.0, "cldice": 0.5}


def loss_terms(names) -> tuple[str, ...]:
    """The terms that `names` (--loss: terms and aliases) stand for, each once, in LOSS_TERMS order."""
    chosen = set()
    for name in [names] if isinstance(names, str) else names:
        for term in LOSS_ALIASES.get(name, (name,)):
            if term not in LOSS_TERMS:
                raise ValueError(f"Unknown loss term {term!r}: choose from {LOSS_TERMS} or {tuple(LOSS_ALIASES)}.")
            chosen.add(term)
    if not chosen:
        raise ValueError("The loss needs at least one term.")
    return tuple(term for term in LOSS_TERMS if term in chosen)


def loss_label(names) -> str:
    """'BCE + Tversky + clDice' for ('bce', 'tversky', 'cldice')."""
    return " + ".join(TERM_LABELS[term] for term in loss_terms(names))


class CrackLoss(nn.Module):
    """A weighted sum of loss terms on the crack logits, valid-pixel masked.

    `terms` names the terms (LOSS_TERMS; an alias of LOSS_ALIASES is expanded) and `weights`
    their weights (DEFAULT_TERM_WEIGHTS otherwise):

    * ``bce``: binary cross-entropy, averaged over the supervised pixels.
    * ``focal``: binary focal loss (Lin et al., 2017) with `focal_alpha` and `focal_gamma`.
    * ``dice``: 1 - soft Dice over the batch.
    * ``tversky``: 1 - Tversky index over the batch, with `false_positive_weight` and
      `false_negative_weight` (0.4 and 0.6 lean towards the recall of faint cracks).
    * ``cldice``: soft clDice (Shit et al., 2021), scaled by `cldice_scale` for its warm-up.

    `label_tolerance` > 0 leaves the ring of that many pixels around every label
    unsupervised. Labels drawn by hand (the DXF tiles) sit a few pixels off the
    crack, so a strict loss punishes a prediction on the crack itself as a false
    positive plus a miss; the tol_* metrics already accept a prediction that close.

    `outputs` is the crack logits, or ``(logits, auxiliary)`` or ``(logits, auxiliary, distance)``
    as HRCrackNet returns them in training: the deep-supervision logits (1/4 and 1/2 scale),
    weighted by `auxiliary_weights` and trained on the region terms alone, and the logits of the
    distance-field head (or None), trained with `distance_weight` times ``distance_loss`` against
    the field of the mask up to `distance_margin` pixels. After a call `output_term` holds the
    detached loss of the network output without those terms, which is what the validation loss
    measures, and `distance_term` the detached distance loss (0 without a distance output).
    """

    def __init__(
        self, terms=LOSS_ALIASES["hairline"], weights=None, false_positive_weight=0.4, false_negative_weight=0.6,
        focal_alpha=0.25, focal_gamma=2.0, skeleton_iterations=10, auxiliary_weights=(0.2, 0.2),
        label_tolerance=0, distance_weight=0.5, distance_margin=5,
    ):
        super().__init__()
        self.terms = loss_terms(terms)
        weights = {**DEFAULT_TERM_WEIGHTS, **(weights or {})}
        self.weights = {term: float(weights[term]) for term in self.terms}
        self.fp_weight, self.fn_weight = false_positive_weight, false_negative_weight
        self.alpha, self.gamma = focal_alpha, focal_gamma
        self.skeleton_iterations = skeleton_iterations
        self.auxiliary_weights = tuple(auxiliary_weights)
        self.label_tolerance = int(label_tolerance)
        self.distance_weight, self.distance_margin = float(distance_weight), int(distance_margin)
        self.output_term, self.distance_term = None, None

    def region_loss(self, logits, target, valid):
        """The region terms: returns (loss, probability * valid, target * valid)."""
        weights = self.weights
        logits = logits.float()
        loss = logits.new_zeros(())
        if "bce" in weights or "focal" in weights:
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            supervised = valid.sum().clamp_min(1.0)
            if "bce" in weights:
                loss = loss + weights["bce"] * (bce * valid).sum() / supervised
            if "focal" in weights:
                probability = torch.sigmoid(logits)
                # p_t is the probability given to the true class: (1 - p_t)^gamma turns the loss down on
                # pixels that are already right, which is nearly all of the background.
                true_class = probability * target + (1 - probability) * (1 - target)
                class_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
                focal = class_weight * (1 - true_class) ** self.gamma * bce
                loss = loss + weights["focal"] * (focal * valid).sum() / supervised
        probability, target = torch.sigmoid(logits) * valid, target * valid
        if "dice" in weights or "tversky" in weights:
            true_positive = (probability * target).sum()
            false_positive = (probability * (1 - target)).sum()
            false_negative = ((1 - probability) * target).sum()
            if "dice" in weights:
                dice = (2 * true_positive + 1.0) / (2 * true_positive + false_positive + false_negative + 1.0)
                loss = loss + weights["dice"] * (1 - dice)
            if "tversky" in weights:
                tversky = (true_positive + 1.0) / (
                    true_positive + self.fp_weight * false_positive + self.fn_weight * false_negative + 1.0
                )
                loss = loss + weights["tversky"] * (1 - tversky)
        return loss, probability, target

    def forward(self, outputs, target, valid, cldice_scale: float = 1.0):
        logits, auxiliary, distance = (*outputs, None)[:3] if isinstance(outputs, tuple) else (outputs, (), None)
        field_valid = valid  # the distance field needs no tolerance ring: it is graded around the label anyway
        if self.label_tolerance > 0:
            ring = F.max_pool2d(target, 2 * self.label_tolerance + 1, 1, self.label_tolerance) - target
            valid = valid * (1 - ring)
        loss, probability, target_valid = self.region_loss(logits, target, valid)
        weight = self.weights.get("cldice", 0.0) * cldice_scale
        if weight > 0:
            loss = loss + weight * soft_cldice_loss(probability, target_valid, self.skeleton_iterations)
        self.output_term = loss.detach()
        for auxiliary_weight, auxiliary_logits in zip(self.auxiliary_weights, auxiliary):
            loss = loss + auxiliary_weight * self.region_loss(auxiliary_logits, target, valid)[0]
        self.distance_term = loss.new_zeros(())
        if distance is not None and self.distance_weight > 0:
            self.distance_term = distance_loss(distance, target, field_valid, self.distance_margin)
            loss = loss + self.distance_weight * self.distance_term
            self.distance_term = self.distance_term.detach()
        return loss


class HairlineCrackLoss(CrackLoss):
    """BCE + Tversky (region) + clDice (centreline continuity): the loss of the other hairline trainers."""

    def __init__(
        self, bce_weight=1.0, tversky_weight=1.0, cldice_weight=0.5,
        false_positive_weight=0.4, false_negative_weight=0.6,
        skeleton_iterations=10, auxiliary_weights=(0.2, 0.2), label_tolerance=0, distance_weight=0.5,
        distance_margin=5,
    ):
        super().__init__(LOSS_ALIASES["hairline"], {"bce": bce_weight, "tversky": tversky_weight,
                                                    "cldice": cldice_weight},
                         false_positive_weight, false_negative_weight, skeleton_iterations=skeleton_iterations,
                         auxiliary_weights=auxiliary_weights, label_tolerance=label_tolerance,
                         distance_weight=distance_weight, distance_margin=distance_margin)


class FocalCrackLoss(CrackLoss):
    """Binary focal loss (Lin et al., 2017) + clDice: HairlineCrackLoss with the focal loss as region term."""

    def __init__(self, alpha=0.25, gamma=2.0, cldice_weight=0.5, skeleton_iterations=10,
                 auxiliary_weights=(0.2, 0.2), label_tolerance=0, distance_weight=0.5, distance_margin=5):
        super().__init__(("focal", "cldice"), {"focal": 1.0, "cldice": cldice_weight}, focal_alpha=alpha,
                         focal_gamma=gamma, skeleton_iterations=skeleton_iterations,
                         auxiliary_weights=auxiliary_weights, label_tolerance=label_tolerance,
                         distance_weight=distance_weight, distance_margin=distance_margin)


__all__ = [
    'soft_erode', 'soft_skeleton', 'soft_cldice_loss', 'distance_field', 'distance_loss', 'LOSS_TERMS', 'REGION_TERMS', 'LOSS_ALIASES', 'TERM_LABELS',
    'DEFAULT_TERM_WEIGHTS', 'loss_terms', 'loss_label', 'CrackLoss', 'HairlineCrackLoss', 'FocalCrackLoss',
]
