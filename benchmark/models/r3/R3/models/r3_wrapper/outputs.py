"""Output formatting helpers for the R3 wrapper."""

import addict
import torch

from depth_anything_3.model.utils.transform import (
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)
from depth_anything_3.utils.alignment import compute_sky_mask as compute_non_sky_mask
from depth_anything_3.utils.geometry import affine_inverse
from R3.models.online.pose_resolution import resolve_reconstructed_pose_enc


class R3OutputMixin:
    @staticmethod
    def _get_output_value(output, key, default=None):
        if isinstance(output, dict):
            return output.get(key, default)
        return getattr(output, key, default)

    def _format_predictions(
        self,
        output,
        images,
        H,
        W,
        rel_pose_reconstruction_method,
        rel_pose_reconstruction_kwargs,
        online_mode: bool = False,
        rel_pose_frame_ids=None,
        anchored_pose_enc=None,
    ):
        predictions = {}

        rel_pose_enc = self._get_output_value(output, "rel_pose_enc")
        rel_pose_conf = self._get_output_value(output, "rel_pose_conf")
        rel_pose_conf_t = self._get_output_value(output, "rel_pose_conf_t")
        rel_pose_conf_r = self._get_output_value(output, "rel_pose_conf_r")
        rel_pose_mask = self._get_output_value(output, "rel_pose_mask")
        has_rel_pose_predictions = (
            isinstance(rel_pose_enc, torch.Tensor)
            and isinstance(rel_pose_conf, torch.Tensor)
            and isinstance(rel_pose_mask, torch.Tensor)
        )

        if "depth" in output and not isinstance(output.depth, addict.Dict):
            depth = output.depth
            predictions["depth"] = depth.unsqueeze(-1)
        else:
            predictions["depth"] = None

        if "depth_conf" in output and not isinstance(output.depth_conf, addict.Dict):
            predictions["depth_conf"] = output.depth_conf
        else:
            predictions["depth_conf"] = None

        sky = self._get_output_value(output, "sky")
        if isinstance(sky, torch.Tensor):
            predictions["sky"] = sky
            if getattr(self, "compute_sky_mask_enabled", False):
                non_sky_mask = compute_non_sky_mask(
                    sky, threshold=getattr(self, "sky_mask_threshold", 0.3)
                )
                predictions["non_sky_mask"] = non_sky_mask
                predictions["sky_mask"] = ~non_sky_mask

        # In online inference, rel-pose reconstruction provides pose_enc.
        # Skip expensive pose list conversion when those predictions exist.
        should_convert_pose_lists = not (
            online_mode and not self.training and has_rel_pose_predictions
        )
        if (
            should_convert_pose_lists
            and "extrinsics" in output
            and "intrinsics" in output
        ):
            predictions["pose_enc_list"] = []
            if "pose_enc_list" in output:
                for pe in output.pose_enc_list:
                    c2w_i, ixt_i = pose_encoding_to_extri_intri(pe, (H, W))
                    w2c_i = affine_inverse(c2w_i)
                    pe_w2c = extri_intri_to_pose_encoding(w2c_i, ixt_i, (H, W))
                    predictions["pose_enc_list"].append(pe_w2c)
                predictions["pose_enc"] = predictions["pose_enc_list"][-1]

            if "pose_enc_list_aux" in output:
                predictions["pose_enc_list_aux"] = []
                for pe in output.pose_enc_list_aux:
                    c2w_i, ixt_i = pose_encoding_to_extri_intri(pe, (H, W))
                    w2c_i = affine_inverse(c2w_i)
                    pe_w2c = extri_intri_to_pose_encoding(w2c_i, ixt_i, (H, W))
                    predictions["pose_enc_list_aux"].append(pe_w2c)

            predictions["world_points"] = None
            predictions["world_points_conf"] = None

        if has_rel_pose_predictions:
            predictions["rel_pose_enc_list"] = [rel_pose_enc]
            predictions["rel_pose_conf_list"] = [rel_pose_conf]
            if isinstance(rel_pose_conf_t, torch.Tensor):
                predictions["rel_pose_conf_t_list"] = [rel_pose_conf_t]
            if isinstance(rel_pose_conf_r, torch.Tensor):
                predictions["rel_pose_conf_r_list"] = [rel_pose_conf_r]
            predictions["rel_pose_mask_list"] = [rel_pose_mask]
            if not self.training:
                # print(
                #     "Using relative pose predictions for online inference, mode:",
                #     rel_pose_reconstruction_method,
                # )
                predictions["pose_enc"] = self._resolve_pose_predictions(
                    predictions,
                    rel_pose_enc,
                    rel_pose_conf,
                    rel_pose_conf_t,
                    rel_pose_conf_r,
                    rel_pose_mask,
                    rel_pose_reconstruction_method,
                    rel_pose_reconstruction_kwargs,
                    online_mode=online_mode,
                    rel_pose_frame_ids=rel_pose_frame_ids,
                    anchored_pose_enc=anchored_pose_enc,
                )

        if hasattr(output, "aux"):
            aux = output.aux
            if (
                self.teacher_embed_dim is not None
                and self.student_embed_dim is not None
                and self.teacher_embed_dim != self.student_embed_dim
            ):
                raise NotImplementedError(
                    "Feature projection for distillation is not implemented yet"
                )
            else:
                predictions["aux"] = aux

        scale_pred = self._get_output_value(output, "scale_pred")
        if isinstance(scale_pred, torch.Tensor):
            predictions["scale_pred"] = scale_pred

        if not self.training:
            predictions["images"] = images

        return predictions

    def _resolve_pose_predictions(
        self,
        predictions,
        rel_pose_enc,
        rel_pose_conf,
        rel_pose_conf_t,
        rel_pose_conf_r,
        rel_pose_mask,
        rel_pose_reconstruction_method,
        rel_pose_reconstruction_kwargs,
        online_mode: bool,
        rel_pose_frame_ids=None,
        anchored_pose_enc=None,
    ):
        return resolve_reconstructed_pose_enc(
            predictions,
            rel_pose_enc,
            rel_pose_conf,
            rel_pose_mask,
            rel_pose_reconstruction_method,
            rel_pose_reconstruction_kwargs,
            rel_pose_conf_t=rel_pose_conf_t,
            rel_pose_conf_r=rel_pose_conf_r,
            online_mode=online_mode,
            rel_pose_frame_ids=rel_pose_frame_ids,
            anchored_pose_enc=anchored_pose_enc,
        )

    def _merge_online_step_predictions(self, predictions_list):
        merged_predictions = {}
        for predictions in predictions_list:
            for key, value in predictions.items():
                if value is None:
                    continue
                if key in {
                    "depth",
                    "depth_conf",
                    "pose_enc",
                    "images",
                    "sky",
                    "sky_mask",
                    "non_sky_mask",
                }:
                    merged_predictions[key] = torch.cat(
                        [merged_predictions.get(key, value[:, :0]), value], dim=1
                    )
                elif key in {
                    "pose_enc_list",
                    "pose_enc_list_aux",
                    "rel_pose_enc_list",
                    "rel_pose_conf_list",
                    "rel_pose_conf_t_list",
                    "rel_pose_conf_r_list",
                    "rel_pose_mask_list",
                    "rel_pose_frame_ids_list",
                }:
                    merged_predictions.setdefault(key, []).extend(value)
                elif key == "pose_enc_pool_local":
                    merged_predictions.setdefault(key, []).append(value)
                elif key == "rel_pose_frame_ids":
                    merged_predictions.setdefault("rel_pose_frame_ids_list", []).append(
                        value
                    )
                elif key == "output_frame_ids":
                    merged_predictions.setdefault(key, []).extend(value)
                else:
                    merged_predictions[key] = value
        return merged_predictions
