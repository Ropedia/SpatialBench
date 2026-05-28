"""Setup helpers for the R3 wrapper."""

import types

import addict

from depth_anything_3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net


class R3SetupMixin:
    def _get_backbone_model(self):
        backbone_model = getattr(self.da3, "backbone", None)
        if backbone_model is not None and hasattr(backbone_model, "pretrained"):
            backbone_model = backbone_model.pretrained
        return backbone_model

    def _uses_causal_attention(self):
        backbone_model = self._get_backbone_model()
        return bool(getattr(backbone_model, "causal_attn", False))

    def set_freeze(self, freeze: str):
        if not freeze or freeze == "none":
            return

        print(f"Applying freeze strategy: {freeze}")

        # 1. Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        # 2. Parse modes (supports comma-separated list)
        modes = [m.strip() for m in freeze.split(",")]

        # Helper to get the actual backbone model
        backbone_model = None
        if hasattr(self.da3, "backbone"):
            backbone_model = self.da3.backbone
            # Handle the case where backbone wraps the actual model in .pretrained (e.g. DinoV2 wrapper)
            if hasattr(backbone_model, "pretrained"):
                backbone_model = backbone_model.pretrained

        num_blocks = (
            len(backbone_model.blocks)
            if backbone_model is not None and hasattr(backbone_model, "blocks")
            else 0
        )
        if "backbone" in modes:
            print("  Unfreezing full backbone")
            for param in self.da3.backbone.parameters():
                param.requires_grad = True
        else:
            # selective backbone unfreezing
            if "global" in modes:
                print("  Unfreezing entire global backbone blocks")
                unfreeze_idx = [idx for idx in range(8, num_blocks) if idx % 2 == 1]
                if backbone_model and hasattr(backbone_model, "blocks"):
                    for idx in unfreeze_idx:
                        if idx < len(backbone_model.blocks):
                            for param in backbone_model.blocks[idx].parameters():
                                param.requires_grad = True
                    # Re-freeze LayerScale gammas in global blocks — their pretrained values
                    # are large (up to ~17) and training them causes gradient explosion.

            if "local" in modes:
                print("  Unfreezing local backbone blocks")
                unfreeze_idx = [idx for idx in range(8, num_blocks) if idx % 2 == 0]
                if backbone_model and hasattr(backbone_model, "blocks"):
                    for idx in unfreeze_idx:
                        if idx < len(backbone_model.blocks):
                            for param in backbone_model.blocks[idx].parameters():
                                param.requires_grad = True

            if "linear" in modes:
                print("  Unfreezing extra linear blocks")
                if (
                    backbone_model
                    and hasattr(backbone_model, "extra_blocks")
                    and backbone_model.extra_blocks is not None
                ):
                    for param in backbone_model.extra_blocks.parameters():
                        param.requires_grad = True

        # --- Camera Freeze Logic ---
        unfreeze_cam_dec = "cam_dec" in modes
        unfreeze_cam_enc = "cam_enc" in modes

        if unfreeze_cam_dec:
            print("  Unfreezing camera decoder")
            if hasattr(self.da3, "cam_dec") and self.da3.cam_dec is not None:
                for param in self.da3.cam_dec.parameters():
                    param.requires_grad = True
                # Re-freeze absolute-only heads when using relative camera —
                # fc_t and fc_qvec only feed abs_pose_enc (not in rel_cam loss),
                # and fc_abs_fov is dead code (never called in forward).
                if getattr(self.da3, "relative_cam", False):
                    cam_dec = self.da3.cam_dec
                    for module in [
                        getattr(cam_dec, "fc_t", None),
                        getattr(cam_dec, "fc_qvec", None),
                        getattr(cam_dec, "fc_abs_fov", None),
                    ]:
                        if module is not None:
                            for param in module.parameters():
                                param.requires_grad = False
            # Skip aux_cam_decs when using relative camera — their outputs are not
            # included in the loss (only the absolute-camera path uses them), so
            # unfreezing them would create unused parameters that break DDP.
            if hasattr(self.da3, "aux_cam_decs") and not getattr(
                self.da3, "relative_cam", False
            ):
                for cam_dec in self.da3.aux_cam_decs:
                    for param in cam_dec.parameters():
                        param.requires_grad = True

        if unfreeze_cam_enc:
            print("  Unfreezing camera encoder")
            if hasattr(self.da3, "cam_enc") and self.da3.cam_enc is not None:
                for param in self.da3.cam_enc.parameters():
                    param.requires_grad = True

        if "head" in modes:
            print("  Unfreezing depth head")
            if hasattr(self.da3, "head") and self.da3.head is not None:
                for param in self.da3.head.parameters():
                    param.requires_grad = True

        if hasattr(self, "projections"):
            for param in self.projections.parameters():
                param.requires_grad = True

        # Log trainable parameters count
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        total_params_count = sum(p.numel() for p in trainable_params)
        print(f"Number of trainable parameters (tensors): {len(trainable_params)}")
        print(
            f"Total number of trainable parameters (elements): {total_params_count / 1e6:.2f} M"
        )

        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        if len(trainable_names) > 0:
            print(f"Example trainable param: {trainable_names[0]}")

    def _disable_model_depth_head(self, model):
        """Remove depth heads and short-circuit depth forwarding to save memory/compute."""

        def _bind_no_depth(module):
            # Delete heavy head parameters and override depth branch
            if hasattr(module, "head"):
                module.head = None

            def _no_depth(_self, feats, H, W):
                return addict.Dict()

            module._process_depth_head = types.MethodType(_no_depth, module)

        net = model
        if net is None:
            return

        if isinstance(net, DepthAnything3Net):
            _bind_no_depth(net)
        elif isinstance(net, NestedDepthAnything3Net):
            _bind_no_depth(net.da3)
            _bind_no_depth(net.da3_metric)

            def _nested_forward_short(_self, x, *args, **kwargs):
                return _self.da3(x, *args, **kwargs)

            net.forward = types.MethodType(_nested_forward_short, net)
