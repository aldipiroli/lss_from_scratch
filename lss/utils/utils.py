import torch


def get_noise_scheduler(config):
    T = config["MODEL"]["T"]
    beta_1 = config["MODEL"]["beta_1"]
    beta_T = config["MODEL"]["beta_T"]
    scheaduler = torch.linspace(beta_1, beta_T, T)
    return scheaduler


def compute_alpha_bar_t(scheaduler, t):
    assert t >= 0 and t < len(scheaduler), f"t={t} is outside of scheaduler boundaries"
    alpha_bar = 1
    for i in range(t + 1):
        alpha_t = 1 - scheaduler[i]
        alpha_bar *= alpha_t
    return torch.tensor(alpha_bar)


def precompute_alpha_t(config):
    scheaduler = get_noise_scheduler(config)
    all_alpha_bar = []
    for t in range(len(scheaduler)):
        alpha_bar_t = compute_alpha_bar_t(scheaduler, t)
        all_alpha_bar.append(alpha_bar_t)
    all_alpha_bar = torch.stack(all_alpha_bar)
    return all_alpha_bar


def get_noisy_image(img, all_alpha_bar):
    B = img.shape[0]
    t = torch.randint(0, len(all_alpha_bar), (B, 1))
    eps = torch.randn(img.shape).to(img.device)
    alpha_bar_t = all_alpha_bar[t].view(B, 1, 1, 1).to(img.device)
    keep = torch.sqrt(alpha_bar_t) * img
    noise = torch.sqrt(1 - alpha_bar_t) * eps
    img_noisy = keep + noise
    return img_noisy, eps, t


def denoise_image_step(
    x_t, pred_eps_cond, pred_eps_uncond, t, scheaduler, all_alpha_bar, weight=0.1
):
    t_idx = t.item() if isinstance(t, torch.Tensor) else t
    alpha_t = (1 - scheaduler[t_idx]).to(x_t.device)
    alpha_bar_t = all_alpha_bar[t_idx].to(x_t.device)
    beta_t = scheaduler[t_idx].to(x_t.device)
    sigma_t = torch.sqrt(beta_t)
    pred_eps = (1 + weight) * pred_eps_cond - weight * pred_eps_uncond
    z = (
        torch.randn(pred_eps.shape, device=x_t.device)
        if t_idx > 0
        else torch.zeros_like(pred_eps)
    )
    x_t_m1 = (
        1
        / torch.sqrt(alpha_t)
        * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * pred_eps)
        + sigma_t * z
    )
    return x_t_m1


def drop_class_condition(labels, config):
    B = labels.shape[0]
    num_classes = config["MODEL"]["num_classes"]
    class_drop_prob = config["MODEL"]["class_drop_prob"]
    drop_mask = torch.rand(B, device=labels.device) < class_drop_prob
    labels[drop_mask] = num_classes
    return labels
