# RiverGuard labeling guide

RiverGuard raises an alert by **counting boxes**, so how we box things *is* the definition of "6 items".
Every labeler (and every dataset we merge) has to follow the same rules, or the model learns contradictions.

## Classes

| Class | What goes in it | Examples |
|---|---|---|
| `plastic_bottle` | Plastic bottles of any size, whole or crushed | PET water or soda bottles, detergent or shampoo bottles |
| `plastic_bag` | Flexible plastic bags and film | *kresek* carrier bags, garbage bags, woven plastic sacks (*karung*), loose plastic sheets |
| `plastic_package` | Single-use food and product packaging | snack wrappers, sachets, instant-noodle cups, drink cups, styrofoam food boxes, blister packs |
| `other_plastic` | Any other plastic object | sandals, buckets, jerrycans, toys, pipes, broken plastic pieces |
| `trash_patch` | A **region** of trash packed so densely that single items can't be told apart | a carpet of wrappers and bags against a floodgate, a solid mat of bottles behind a trash boom |

**Don't label** things that aren't plastic: glass bottles, metal cans, paper or cardboard, wood, leaves,
water hyacinth, natural foam, animals, boats or people.

> The public datasets merged for training were remapped to these same rules (see `sources.yaml`):
> their cans, cartons, wood and paper boxes are dropped, and generic "plastic" becomes `other_plastic`.

## Drawing boxes

1. **One box per item you can tell apart.** Include items that are partly underwater or hidden behind
   others, and box only the part you can see.
2. **Dense carpets get `trash_patch` boxes instead.** Where trash is packed so tightly that you'd be guessing
   where one item ends and the next begins, draw one `trash_patch` box around that region. Use a few boxes
   if the carpet is irregular, so that little open water is included. **Don't also box the items inside a
   patch.** Separate items outside it still get their own boxes. A handful of touching bottles is *not* a
   patch: if you can count them, box them.
3. **Skip tiny items**: anything under about 8 px on its long side in the full-size image.
4. **Keep boxes tight** around the visible item, not including its reflection or shadow.
5. **When unsure between two plastic classes**, pick the closer one. For counting, a box with the wrong
   class is much better than a missing box.

## The two Roboflow projects

| Project | Split | Purpose |
|---|---|---|
| `riverguard-test` | test | The ruler for measuring v1 and v2. It is **never** used for training or tuning. Label it extra carefully. |
| `riverguard-hard` | train | Hard examples that fix v1's weak spots: whole river photos with dense carpets (`trash_patch`) and scattered items. |

Tier B test images (scenes where only the SAFE/DANGER outcome matters) aren't boxed at all. They get a single
tag, `danger` / `safe` / `none`, in `datasets/tier-b/tags.csv`, and are used only to test the alarm.

A photo, or frames from the same video, must never appear in both projects. The dataset builder also
drops any training image that looks like a test image, as a safety net.

**Aim for about 200 test images**:
- ~40% scenes with more than 5 items (should trigger DANGER)
- ~40% scenes with 1–5 items (SAFE)
- ~20% with no plastic at all (clean water, leaves, foam, boats), to measure false alarms

## Workflow

1. The images arrive **pre-labeled** by the current model as predictions. For each image:
   fix wrong classes, adjust boxes, delete false ones, and add missed ones. *Smart Select* (click an object)
   or *Label Assist* speeds this up.
2. Approve the image once it's correct.
3. When a project is done, **Generate → New Version** with:
   - Preprocessing: *Auto-Orient* only (no resize, since the training script handles sizing)
   - Augmentations: **none** (training applies its own)
   - Don't use *Rebalance splits*
4. Tell Claude the version number. `training/sources.yaml` points at it.
