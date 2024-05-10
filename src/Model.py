import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import tinycudann as tcnn



class ColorNet(nn.Module):
    def __init__(self, config, input_ch=4, geo_feat_dim=15,
                 hidden_dim_color=64, num_layers_color=3):
        super(ColorNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_color = num_layers_color
        self.embed2color = self.config["HashGrid"]["embed2color"]
        self.model = self.get_model(config['HashGrid']['tcnn_network'])

    def forward(self, input_feat):
        return self.model(input_feat)

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
            )

        color_net = []
        for l in range(self.num_layers_color):
            if l == 0:
                if self.embed2color:
                    in_dim = self.input_ch + self.geo_feat_dim
                else:
                    in_dim = 80
            else:
                in_dim = self.hidden_dim_color

            if l == self.num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = self.hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))


class SDFNet(nn.Module):
    def __init__(self, config, input_ch=3, geo_feat_dim=15, hidden_dim=64, num_layers=2):
        super(SDFNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.model = self.get_model(tcnn_network=config['HashGrid']['tcnn_network'])

    def forward(self, x, return_geo=True):
        out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
                # dtype=torch.float
            )
        else:
            sdf_net = []
            for l in range(self.num_layers):
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = self.hidden_dim

                if l == self.num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
                else:
                    out_dim = self.hidden_dim

                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l != self.num_layers - 1:
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))


class HashGrid(nn.Module):
    def __init__(self, config):
        super(HashGrid, self).__init__()
        self.config = config
        self.embed2color = config["HashGrid"]["embed2color"]
        self.dosig = config["HashGrid"]["sigmoid"]
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.config["device"])
        self.get_resolution()

        self.learnable_beta = self.config["sampleAndLoss"]["learnable_beta"]
        if self.learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10

        # Coordinate encoding
        self.embedpos_fn, self.input_ch_pos = get_encoder(config['HashGrid']['posEnc'],
                                                          n_bins=self.config['HashGrid']['n_bins'])
        # Sparse parametric encoding (SDF)
        self.embed_fn, self.input_ch = get_encoder('HashGrid', log2_hashmap_size=config['HashGrid']['hash_size'],
                                                   desired_resolution=self.resolution_sdf)

        self.color_net = ColorNet(config,
                                  input_ch=self.input_ch_pos,
                                  geo_feat_dim=config['HashGrid']['geo_feat_dim'],
                                  hidden_dim_color=config['HashGrid']['hidden_dim_color'],
                                  num_layers_color=config['HashGrid']['num_layers_color'])
        self.sdf_net = SDFNet(config,
                              input_ch=self.input_ch + self.input_ch_pos,
                              geo_feat_dim=config['HashGrid']['geo_feat_dim'],
                              hidden_dim=config['HashGrid']['hidden_dim'],
                              num_layers=config['HashGrid']['num_layers'])

    def get_resolution(self):
        '''
        Get the resolution of the grid
        '''
        dim_max = (self.bounding_box[:, 1] - self.bounding_box[:, 0]).max()
        if self.config['HashGrid']['voxel_sdf'] > 10:
            self.resolution_sdf = self.config['HashGrid']['voxel_sdf']
        else:
            self.resolution_sdf = int(dim_max / self.config['HashGrid']['voxel_sdf'])

        if self.config['HashGrid']['voxel_color'] > 10:
            self.resolution_color = self.config['HashGrid']['voxel_color']
        else:
            self.resolution_color = int(dim_max / self.config['HashGrid']['voxel_color'])
    def get_optParameter(self):
        optParameter = [{'params': self.color_net.parameters(), 'weight_decay': 1e-6,
                         'lr': self.config["HashGrid"]["decoder_lr"]},
                        {'params': self.sdf_net.parameters(), 'weight_decay': 1e-6,
                         'lr': self.config["HashGrid"]["decoder_lr"]},
                        {'params': self.embed_fn.parameters(), 'eps': 1e-15,
                         'lr': self.config["HashGrid"]["grid_lr"]}]
        return optParameter

    def get_embedFromGrid(self, points):
        inputs_flat = (points - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        embedded = self.embed_fn(inputs_flat)
        return embedded

    def forward(self, points, onlySdf=False):
        # print(embed.shape, embed_pos.shape)
        inputs_flat = (points - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        embed = self.embed_fn(inputs_flat)
        embed_pos = self.embedpos_fn(inputs_flat)

        h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True)
        sdf, geo_feat = h[..., :1], h[..., 1:]
        if onlySdf:
            return sdf
        if self.embed2color:
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([embed_pos, embed], dim=-1))

        if self.dosig:
            rgb = torch.sigmoid(rgb)

        return torch.cat([rgb, sdf], -1)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.device = self.config["device"]
        bound_dividable = 0.24
        self.bounding_box = torch.from_numpy(np.array(config['mapping']['bound'], dtype=np.float32)).to(
            config['device'])
        self.bounding_box[:, 1] = (((self.bounding_box[:, 1] - self.bounding_box[:, 0]) /
                                    bound_dividable).int() + 1) * bound_dividable + self.bounding_box[:, 0]

        self.decoder = HashGrid(config).to(self.device)

    def sdf2weights(self, sdf):
        beta = self.decoder.beta
        alpha = 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.config["device"])
                                                      , (1. - alpha + 1e-10)], -1), -1)[:, :-1]
        return weights

    def raw2outputs(self, raw, z_vals):

        rgb = raw[..., :3]  # [N_rays, N_samples, 3]
        weights = self.sdf2weights(raw[..., 3])
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)

        return rgb_map, depth_map, raw[..., -1], z_vals

    def query_color_sdf(self, query_points):
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
        return self.decoder(inputs_flat)

    def query_color(self, query_points):
        return self.query_color_sdf(query_points)[..., :3]

    def query_sdf(self, query_points):
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
        return self.decoder.forward(inputs_flat, onlySdf=True)

    def render_rays(self, rays_o, rays_d, target_d):
        truncation = self.config['sampleAndLoss']['truncation']
        n_stratified = self.config['sampleAndLoss']['n_stratified']
        n_importance = self.config['sampleAndLoss']['n_importance']

        n_rays = rays_o.shape[0]
        gt_depth = target_d.squeeze(1)

        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=self.device)
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=self.device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=self.device)

        ### pixels with gt depth:
        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]

        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * truncation) + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
        z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero

        ### pixels without gt depth (importance sampling):
        if not gt_mask.all():
            with torch.no_grad():
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bounding_box.unsqueeze(0) - det_rays_o) / det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                z_vals_uni = self.perturbation(z_vals_uni)
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(
                    -1)  # [n_rays, n_stratified, 3]

                inputs_flat = torch.reshape(pts_uni.clone(), [-1, pts_uni.clone().shape[-1]])
                sdf_uni = self.decoder.forward(inputs_flat)[:, 3]
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = 1. - torch.exp(
                    -self.decoder.beta * torch.sigmoid(-sdf_uni * self.decoder.beta))
                weights_uni = alpha_uni * torch.cumprod(
                    torch.cat([torch.ones((alpha_uni.shape[0], 1), device=self.device)
                                  , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]

                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False,
                                           device=self.device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]
        inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat)
        raw = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
        rgb_map, depth_map, sdf, z_vals = self.raw2outputs(raw, z_vals)
        return rgb_map, depth_map, sdf, z_vals

    def forward(self, rays_o, rays_d, target_rgb, target_d, tracker=False, smooth=False):
        if not hasattr(self, "fs_weight_t"):
            self.fs_weight_t = self.config["tracking"]["fs_weight"]
            self.center_weight_t = self.config["tracking"]["center_weight"]
            self.tail_weight_t = self.config["tracking"]["tail_weight"]
            self.depth_weight_t = self.config["tracking"]["depth_weight"]
            self.color_weight_t = self.config["tracking"]["color_weight"]

            self.fs_weight_m = self.config["mapping"]["fs_weight"]
            self.center_weight_m = self.config["mapping"]["center_weight"]
            self.tail_weight_m = self.config["mapping"]["tail_weight"]
            self.depth_weight_m = self.config["mapping"]["depth_weight"]
            self.color_weight_m = self.config["mapping"]["color_weight"]

            self.fs_m = self.config["sampleAndLoss"]["fs_weight"]
            self.sdf_m = self.config["sampleAndLoss"]["sdf_weight"]
            self.rgb_m = self.config["sampleAndLoss"]["rgb_weight"]
            self.depth_m = self.config["sampleAndLoss"]["depth_weight"]

        rgb_map, depth_map, sdf, z_vals = self.render_rays(rays_o, rays_d, target_d)
        depth_mask = (target_d > 0).squeeze(1)
        fs_loss, center_loss, tail_loss = self.sdf_losses(sdf, z_vals,
                                                          target_d, depth_mask)
        depthLoss = torch.square(target_d.squeeze()[depth_mask] - depth_map[depth_mask]).mean()
        if tracker:
            colorLoss = 0
            loss = self.fs_weight_t * fs_loss + self.center_weight_t * center_loss + self.tail_weight_t * tail_loss + self.depth_weight_t * depthLoss + self.color_weight_t * colorLoss
        else:
            colorLoss = torch.square(target_rgb - rgb_map).mean()
            loss = self.fs_weight_m * fs_loss + self.center_weight_m * center_loss + self.tail_weight_m * tail_loss + self.depth_weight_m * depthLoss + self.color_weight_m * colorLoss
        return loss

    def perturbation(self, z_vals):
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def sdf_losses(self, sdf, z_vals, gt_depth, depth_mask):
        self.truncation = self.config["sampleAndLoss"]["truncation"]
        trunc_factor = self.config["sampleAndLoss"]["trunc_factor"]
        sdf = sdf[depth_mask]
        z_vals = z_vals[depth_mask]
        gt_depth = gt_depth[depth_mask]
        front_mask = torch.where(z_vals < (gt_depth - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        back_mask = torch.where(z_vals > (gt_depth + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        center_mask = torch.where((z_vals > (gt_depth - trunc_factor * self.truncation)) *
                                  (z_vals < (gt_depth + trunc_factor * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)
        fs_loss = F.mse_loss(sdf[front_mask], torch.ones_like(sdf[front_mask]))
        center_loss = F.mse_loss((z_vals + sdf * self.truncation)[center_mask],
                                 gt_depth.expand(z_vals.shape)[center_mask])
        tail_loss = F.mse_loss((z_vals + sdf * self.truncation)[tail_mask],
                               gt_depth.expand(z_vals.shape)[tail_mask])
        return fs_loss, center_loss, tail_loss


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    pdf = weights

    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def coordinates(voxel_dim, device: torch.device, flatten=True):
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)

    if not flatten:
        return torch.stack([x, y, z], dim=-1)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def batchify(fn, chunk=1024 * 64):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, inputs_dir=None):
        if inputs_dir is not None:
            return torch.cat(
                [fn(inputs[i:i + chunk], inputs_dir[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


# same as Co-slam
def get_encoder(encoding, input_dim=3,
                n_bins=16, n_levels=16, level_dim=2,
                base_resolution=16, log2_hashmap_size=19,
                desired_resolution=512):
    # Sparse grid encoding
    if 'hash' in encoding.lower() or 'tiled' in encoding.lower():
        per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": 'HashGrid',
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims
        # print(out_dim)
    # OneBlob encoding
    elif 'blob' in encoding.lower():
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "OneBlob",  # Component type.
                "n_bins": n_bins
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    return embed, out_dim
