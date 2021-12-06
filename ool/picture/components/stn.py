import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


class InvSTN(nn.Module):
    """Inverts the extraction theta (as opposed to placement)"""
    def __init__(self, img_size=(1, 50, 50), patch_size=(1, 20, 20)):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

    def expand(self, z_where):
        # [s,x,y] -> [[s,0,x],
        #             [0,s,y]]
        device = z_where.device
        if z_where.size(1) == 3:
            n = z_where.size(0)
            expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3]).to(device)  # 0-dim is batch
            out = torch.cat((torch.zeros([1, 1], device=device).expand(n, 1), z_where), 1)
            return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)
        return z_where  # Assume it was alrady transformed

    def place(self, z_where, obj_img, patch_size=None, img_size=None):
        """Place transform obj_img onto img of size self.img_size by z_where"""
        patch_size = patch_size if patch_size else self.patch_size
        img_size = img_size if img_size else self.img_size
        n = obj_img.size(0)
        c = obj_img.size(1)
        theta = self.expand(z_where)
        grid = F.affine_grid(theta, torch.Size((n, *img_size)), align_corners=False)
        out = F.grid_sample(obj_img.view(n, *patch_size), grid, align_corners=False)
        return out.view(n, c, *img_size[1:])

    def forward(self, z_where, patch):
        return self.place(z_where, patch)

    def inv_where(self, z_where):
        n = z_where.size(0)
        device = z_where.device
        # Return [s,x,y] -> [1/s,-x/s,-y/s]
        mask = torch.tensor([0., 1., 1.], device=device).expand(n, -1)
        # [s,x,y] -> [1, -x, -y]
        inv_z_where = torch.tensor([1., 0., 0.], device=device).expand(n, -1) - mask * z_where
        # [1, -x, -y] -> [1/s,-x/s,-y/s]
        inv_z_where /= z_where[:, 0:1]
        return inv_z_where

    def extract(self, img, z_where, patch_size=None, img_size=None):
        """Pick out a patch defined by z_where from img"""
        n = img.size(0)
        patch_size = patch_size if patch_size else self.patch_size
        img_size = img_size if img_size else self.img_size
        theta_inv = self.expand(self.inv_where(z_where))
        grid = F.affine_grid(theta_inv, (n, *patch_size), align_corners=False)
        out = F.grid_sample(img.view(n, *img_size), grid, align_corners=False)
        return out.view(n, *self.patch_size)

    def bbox(self, z_where):
        """Calculate a bbox based on z_where and img_shape"""
        margin = 1
        s, x, y = z_where[:, 0:1], z_where[:, 1:2], z_where[:, 2:]
        h = self.img_size[1] / s
        w = self.img_size[2] / s

        xtrans = -x / s * self.img_size[2] / 2
        ytrans = -y / s * self.img_size[1] / 2
        x1 = (self.img_size[2] - w) / 2 + xtrans - margin
        y1 = (self.img_size[1] - h) / 2 + ytrans - margin
        x2 = x1 + w + 2 * margin
        y2 = y1 + h + 2 * margin
        xs = torch.hstack([x1, x2])
        x1_t = torch.min(xs, dim=-1)[0].view(-1, 1)
        x2_t = torch.max(xs, dim=-1)[0].view(-1, 1)
        ys = torch.hstack([y1, y2])
        y1_t = torch.min(ys, dim=-1)[0].view(-1, 1)
        y2_t = torch.max(ys, dim=-1)[0].view(-1, 1)
        bbox = torch.hstack([x1_t, y1_t, x2_t, y2_t])
        clipped_bbox = tv.ops.clip_boxes_to_image(bbox, self.img_size[1:])
        return clipped_bbox


class STN:
    def fix_shape(self, shape, c=None):
        if len(shape) == 3:
            c = c | shape[0]
            return (c, *shape[1:])
        if len(shape) == 2:
            if c is None:
                raise ValueError(f"Missing channel spec")
            return (c, *shape)
        raise ValueError(f"Unknown shape {shape}")

    def maybe_fix_z_where(self, z_where):
        if z_where.size(1) == 3:
            s = z_where[:, :1]
            z_where = torch.hstack([s, z_where])
        return z_where

    def inv_where(self, z_where):
        n = z_where.size(0)
        device = z_where.device
        # Return [sx,sy,x,y] -> [1/sx,1/sy,-x/sx,-y/sy]
        mask = torch.tensor([0., 0., 1., 1.], device=device).expand(n, -1)
        # [sx,sy,x,y] -> [1, 1, -x, -y]
        inv_z_where = torch.tensor([1., 1., 0., 0.], device=device).expand(n, -1) - mask * z_where
        # [1, 1, -x, -y] -> [1/s,-x/s,-y/s]
        inv_z_where = inv_z_where / torch.hstack([z_where[:, :2]] * 2)
        return inv_z_where

    def expand(self, z_where):
        # [sx,sy,x,y] -> [[sx,0,x],
        #                 [0,sy,y]]
        device = z_where.device
        if z_where.size(1) == 4:
            n = z_where.size(0)
            expansion_indices = torch.LongTensor([1, 0, 3, 0, 2, 4]).to(device)  # 0-dim is batch
            out = torch.cat((torch.zeros([1, 1], device=device).expand(n, 1), z_where), 1)
            return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)
        return z_where  # Assume it was alrady transformed

    def place(self, z_where, obj, img_size):
        """Place transform obj_img onto img of size self.img_size by z_where"""
        n = obj.size(0)
        c = obj.size(1)
        z_where = self.maybe_fix_z_where(z_where)
        img_size = self.fix_shape(img_size, c)
        theta = self.expand(self.inv_where(z_where))
        grid = F.affine_grid(theta, torch.Size((n, *img_size)), align_corners=False)
        out = F.grid_sample(obj, grid, align_corners=False)
        return out.view(n, *img_size)

    def extract(self, img, z_where, patch_size):
        """Pick out a patch defined by z_where from img"""
        n = img.size(0)
        c = img.size(1)
        patch_size = self.fix_shape(patch_size, c)
        z_where = self.maybe_fix_z_where(z_where)
        theta_inv = self.expand(z_where)
        grid = F.affine_grid(theta_inv, (n, *patch_size), align_corners=False)
        out = F.grid_sample(img, grid, align_corners=False)
        return out.view(n, *patch_size)

    def bbox(self, z_where, wh_shape=None):
        z_where = self.maybe_fix_z_where(z_where)
        sx, sy, x, y = z_where[:, 0:1], z_where[:, 1:2], z_where[:, 2:3], z_where[:, 3:]
        x = (x + 1) / 2 - sx / 2
        y = (1 + y) / 2 - sy / 2
        ret = torch.hstack([x, y, x + sx, y + sy])
        if wh_shape is not None:
            if not isinstance(wh_shape, torch.Tensor):
                wh_shape = self.fix_shape(wh_shape, c=1)[1:]
                sh = torch.tensor(wh_shape)
                clip_shape = tuple(wh_shape[::-1])
            else:
                sh = wh_shape
                clip_shape = tuple(wh_shape.squeeze().tolist()[::-1])
            sh = torch.hstack([sh, sh]).unsqueeze(0).to(z_where.device)
            ret = ret * sh
            ret = tv.ops.clip_boxes_to_image(ret, clip_shape)
        return ret
