from IPython import embed
from abc import abstractmethod
import torch
from tqdm.auto import tqdm
from IPython import embed
from attacks.Attack import Attack
import numpy as np


class Square(Attack):
    def __init__(self, model, model_config, attack_config):
        super().__init__(model, model_config, attack_config)

    # def attack_untargeted(self, x, y):
    #     dim = torch.prod(torch.tensor(x.shape[1:]))
    #
    #     def p_selection(step):
    #         step = int(step / self.attack_config["max_iter"] * 10000)
    #         if 10 < step <= 50:
    #             p = self.attack_config["p_init"] / 2
    #         elif 50 < step <= 200:
    #             p = self.attack_config["p_init"] / 4
    #         elif 200 < step <= 500:
    #             p = self.attack_config["p_init"] / 8
    #         elif 500 < step <= 1000:
    #             p = self.attack_config["p_init"] / 16
    #         elif 1000 < step <= 2000:
    #             p = self.attack_config["p_init"] / 32
    #         elif 2000 < step <= 4000:
    #             p = self.attack_config["p_init"] / 64
    #         elif 4000 < step <= 6000:
    #             p = self.attack_config["p_init"] / 128
    #         elif 6000 < step <= 8000:
    #             p = self.attack_config["p_init"] / 256
    #         elif 8000 < step <= 10000:
    #             p = self.attack_config["p_init"] / 512
    #         else:
    #             p = self.attack_config["p_init"]
    #         return p
    #
    #     def margin_loss(x, y):
    #         logits, is_cache = self.model(x)
    #         probs = torch.softmax(logits, dim=1)
    #         top_2_probs, top_2_classes = torch.topk(probs, 2)
    #         if top_2_classes[:, 0] != y:
    #             return 0, is_cache
    #         else:
    #             return top_2_probs[:, 0] - top_2_probs[:, 1], is_cache
    #
    #     # Initialize adversarial example
    #     pert = torch.tensor(np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]],
    #                                          size=[x.shape[0], x.shape[1], 1, x.shape[3]])).float().to(x.device)
    #     x_adv = torch.clamp(x + pert, 0, 1)
    #     loss, is_cache = margin_loss(x_adv, y)
    #
    #     pbar = tqdm(range(self.attack_config["max_iter"]))
    #     for t in pbar:
    #         x_adv_candidate = x_adv.clone()
    #         for _ in range(self.attack_config["num_squares"]):
    #             # pert = x_adv - x
    #             pert = x_adv_candidate - x
    #             s = int(min(max(torch.sqrt(p_selection(t) * dim / x.shape[1]).round().item(), 1), x.shape[2] - 1))
    #             center_h = torch.randint(0, x.shape[2] - s, size=(1,)).to(x.device)
    #             center_w = torch.randint(0, x.shape[3] - s, size=(1,)).to(x.device)
    #             x_window = x[:, :, center_h:center_h + s, center_w:center_w + s]
    #             x_adv_window = x_adv_candidate[:, :, center_h:center_h + s, center_w:center_w + s]
    #
    #             while torch.sum(
    #                     torch.abs(
    #                         torch.clamp(
    #                             x_window + pert[:, :, center_h:center_h + s, center_w:center_w + s], 0, 1
    #                         ) -
    #                         x_adv_window)
    #                     < 10 ** -7) == x_adv.shape[1] * s * s:
    #                 pert[:, :, center_h:center_h + s, center_w:center_w + s] = torch.tensor(
    #                     np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]], size=[x_adv.shape[1], 1, 1])).float().to(x_adv.device)
    #
    #             x_adv_candidate = torch.clamp(x + pert, 0, 1)
    #         new_loss, is_cache = margin_loss(x_adv_candidate, y)
    #         if is_cache[0]:
    #             continue
    #         if new_loss < loss:
    #             x_adv = x_adv_candidate.clone()
    #             loss = new_loss
    #         pbar.set_description(
    #             f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
    #         if loss == 0:
    #             assert torch.max(torch.abs(x_adv - x)) <= self.attack_config["eps"] + 10 ** -4
    #             return x_adv
    #     return x

    def p_selection(self, step):
        step = int(step / self.attack_config["max_iter"] * 10000)
        if 10 < step <= 50:
            p = self.attack_config["p_init"] / 2
        elif 50 < step <= 200:
            p = self.attack_config["p_init"] / 4
        elif 200 < step <= 500:
            p = self.attack_config["p_init"] / 8
        elif 500 < step <= 1000:
            p = self.attack_config["p_init"] / 16
        elif 1000 < step <= 2000:
            p = self.attack_config["p_init"] / 32
        elif 2000 < step <= 4000:
            p = self.attack_config["p_init"] / 64
        elif 4000 < step <= 6000:
            p = self.attack_config["p_init"] / 128
        elif 6000 < step <= 8000:
            p = self.attack_config["p_init"] / 256
        elif 8000 < step <= 10000:
            p = self.attack_config["p_init"] / 512
        else:
            p = self.attack_config["p_init"]
        return p

    def margin_loss(self, x, y):
        logits, is_cache = self.model(x)
        logits = logits.cpu()
        probs = torch.softmax(logits, dim=1)
        top_2_probs, top_2_classes = torch.topk(probs, 2)
        if top_2_classes[:, 0] != y:
            return 0, is_cache
        else:
            return (top_2_probs[:, 0] - top_2_probs[:, 1]).cpu().numpy(), is_cache

    # def attack_untargeted(self, x, y):
    #     dim = torch.prod(torch.tensor(x.shape[1:]))

    #     # Initialize adversarial example
    #     pert = torch.tensor(np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]],
    #                                          size=[x.shape[0], x.shape[1], 1, x.shape[3]])).float()
    #     x_adv = torch.clamp(x + pert, 0, 1)
    #     loss, is_cache = self.margin_loss(x_adv, y)
    #     if self.attack_config["adaptive"]["bs_num_squares"]:
    #         ns = self.binary_search_num_squares(x, x_adv)
    #     else:
    #         ns = 1
    #     if self.attack_config["adaptive"]["bs_min_square_size"]:
    #         min_s = self.binary_search_min_square_size(x, x_adv, ns)
    #     else:
    #         min_s = 1

    #     pbar = tqdm(range(self.attack_config["max_iter"]))
    #     step_attempts = 0
    #     for t in pbar:
    #         # x_adv_candidate = x_adv.clone()
    #         s = int(min(max(torch.sqrt(self.p_selection(t) * dim / x.shape[1]).round().item(), 1), x.shape[2] - 1))
    #         if self.attack_config["adaptive"]["bs_min_square_size"]:
    #             s = max(s, min_s)
    #         x_adv_candidate = self.add_squares(x, x_adv, s, ns)

    #         step_attempts += 1
    #         new_loss, is_cache = self.margin_loss(x_adv_candidate, y)
    #         if is_cache[0] and step_attempts < self.attack_config["adaptive"]["max_step_attempts"]:
    #             # pbar.set_description(
    #                 # f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
    #             # print("step cache hit")
    #             continue
    #         elif is_cache[0] and step_attempts >= self.attack_config["adaptive"]["max_step_attempts"]:
    #             self.end("Step movement failure.")
    #         step_attempts = 0
    #         if new_loss < loss:
    #             x_adv = x_adv_candidate.clone()
    #             loss = new_loss
    #         pbar.set_description(
    #             f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
    #         if loss == 0:
    #             assert torch.max(torch.abs(x_adv - x)) <= self.attack_config["eps"] + 10 ** -4
    #             return x_adv
    #     return x
    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = np.zeros([x, y])
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
                max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

            counter2[0] -= 1
            counter2[1] -= 1

        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

        return delta


    def meta_pseudo_gaussian_pert(self, s):
        delta = np.zeros([s, s])
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
            if np.random.rand(1) > 0.5: delta = np.transpose(delta)

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, :s // 2] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
            delta[:s // 2, s // 2:] = self.pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

        return delta

    def attack_untargeted(self, x, y):
        dim = torch.prod(torch.tensor(x.shape[1:]))
        eps = self.attack_config["eps"]
        c, h, w = x.shape[1:]
        # Initialize adversarial example
        delta_init = np.zeros(x.shape)
        s = h // 5
        sp_init = (h - s * 5) // 2
        center_h = sp_init + 0
        for counter in range(h // s):
            center_w = sp_init + 0
            for counter2 in range(w // s):
                delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += self.meta_pseudo_gaussian_pert(s).reshape(
                    [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
                center_w += s
            center_h += s

        x_adv = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, 0, 1)

        # pert = torch.tensor(np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]],
                                            #  size=[x.shape[0], x.shape[1], 1, x.shape[3]])).float()
        # x_adv = torch.clamp(x + pert, 0, 1)
        loss, is_cache = self.margin_loss(x_adv.to(torch.float32), y)
        if self.attack_config["adaptive"]["bs_num_squares"]:
            ns = self.binary_search_num_squares(x, x_adv)
        else:
            ns = 1
        if self.attack_config["adaptive"]["bs_min_square_size"]:
            min_s = self.binary_search_min_square_size(x, x_adv, ns)
        else:
            min_s = 1

        pbar = tqdm(range(self.attack_config["max_iter"]))
        step_attempts = 0
        for t in pbar:
            # x_adv_candidate = x_adv.clone()
            delta_curr = (x_adv - x).cpu().numpy()
            s = int(min(max(torch.sqrt(self.p_selection(t) * dim / x.shape[1]).round().item(), 1), x.shape[2] - 1))
            print(s)
            if self.attack_config["adaptive"]["bs_min_square_size"]:
                s = max(s, min_s)
            # x_adv_candidate = self.add_squares(x, x_adv, s, ns)
            if s % 2 == 0:
                s += 1

            s2 = s + 0
            ### window_1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)
            new_deltas_mask = np.zeros(x.shape)
            new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

            ### window_2
            center_h_2 = np.random.randint(0, h - s2)
            center_w_2 = np.random.randint(0, w - s2)
            new_deltas_mask_2 = np.zeros(x.shape)
            new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
            # norms_window_2 = np.sqrt(
            #     np.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, axis=(-2, -1),
            #         keepdims=True))

            ### compute total norm available
            curr_norms_window = np.sqrt(
                np.sum((delta_curr * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
            curr_norms_image = np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True))
            mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
            norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))

            ### create the updates
            new_deltas = np.ones([x.shape[0], c, s, s])
            new_deltas = new_deltas * self.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
            new_deltas *= np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
            old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
            new_deltas += old_deltas
            new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
                np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
            delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
            delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

            x_new = x + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
            x_adv_candidate = np.clip(x_new, 0, 1)
            curr_norms_image = np.sqrt(np.sum((x_adv_candidate - x).cpu().numpy() ** 2, axis=(1, 2, 3), keepdims=True))


            step_attempts += 1
            new_loss, is_cache = self.margin_loss(x_adv_candidate.to(torch.float32), y)
            if is_cache[0] and step_attempts < self.attack_config["adaptive"]["max_step_attempts"]:
                # pbar.set_description(
                    # f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
                # print("step cache hit")
                continue
            elif is_cache[0] and step_attempts >= self.attack_config["adaptive"]["max_step_attempts"]:
                self.end("Step movement failure.")
            step_attempts = 0
            if new_loss <= loss:
                x_adv = x_adv_candidate.clone()
                loss = new_loss
            pbar.set_description(
                f"Step: {t} | True Label: {y} | Predicted Label: {torch.argmax(self._model.model(x_adv.to(torch.float32)))} | Loss: {loss} | square_size: {s} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
            if loss == 0:
                # assert torch.max(torch.abs(x_adv - x)) <= self.attack_config["eps"] + 10 ** -4
                norm_dist = torch.linalg.norm((x_adv - x).to(torch.float32)) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
                print(norm_dist)
                if norm_dist < 0.1:
                    return x_adv.to(torch.float32)
        return x

    def add_squares(self, x, x_adv, s, num_squares):
        x_adv_candidate = x_adv.clone()
        for _ in range(num_squares):
            pert = x_adv_candidate - x

            center_h = torch.randint(0, x.shape[2] - s, size=(1,))
            center_w = torch.randint(0, x.shape[3] - s, size=(1,))
            x_window = x[:, :, center_h:center_h + s, center_w:center_w + s]
            x_adv_window = x_adv_candidate[:, :, center_h:center_h + s, center_w:center_w + s]

            while torch.sum(
                    torch.abs(
                        torch.clamp(
                            x_window + pert[:, :, center_h:center_h + s, center_w:center_w + s], 0, 1
                        ) -
                        x_adv_window)
                    < 10 ** -7) == x.shape[1] * s * s:
                pert[:, :, center_h:center_h + s, center_w:center_w + s] = torch.tensor(
                    np.random.choice([-self.attack_config["eps"], self.attack_config["eps"]],
                                     size=[x.shape[1], 1, 1])).float()
            x_adv_candidate = torch.clamp(x + pert, 0, 1)
        return x_adv_candidate

    def binary_search_num_squares(self, x, x_adv):
        dim = torch.prod(torch.tensor(x.shape[1:]))
        lower = self.attack_config["adaptive"]["bs_num_squares_lower"]
        upper = self.attack_config["adaptive"]["bs_num_squares_upper"]
        ns = upper
        for _ in range(self.attack_config["adaptive"]["bs_num_squares_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_num_squares_sample_size"]):
                s = int(min(max(torch.sqrt(self.p_selection(0) * dim / x.shape[1]).round().item(), 1), x.shape[2] - 1))
                noisy_img = self.add_squares(x, x_adv, s, int(mid))
                probs, is_cache = self.model(noisy_img)
                if is_cache[0]:
                    cache_hits += 1
            if cache_hits / self.attack_config["adaptive"]["bs_num_squares_sample_size"] \
                    <= self.attack_config["adaptive"]["bs_num_squares_hit_rate"]:
                ns = mid
                upper = mid
            else:
                lower = mid
            print(
                f"Num Squares : {ns:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_num_squares_sample_size']}")
        return int(ns)

    def binary_search_min_square_size(self, x, x_adv, num_squares):
        lower = self.attack_config["adaptive"]["bs_min_square_size_lower"]
        upper = self.attack_config["adaptive"]["bs_min_square_size_upper"]
        min_ss = upper
        for _ in range(self.attack_config["adaptive"]["bs_min_square_size_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_min_square_size_sample_size"]):
                noisy_img = self.add_squares(x, x_adv, int(mid), num_squares)
                probs, is_cache = self.model(noisy_img)
                if is_cache[0]:
                    cache_hits += 1
            if cache_hits / self.attack_config["adaptive"]["bs_min_square_size_sample_size"] \
                    <= self.attack_config["adaptive"]["bs_min_square_size_hit_rate"]:
                min_ss = mid
                upper = mid
            else:
                lower = mid
            print(
                f"Min Square Size : {min_ss:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_min_square_size_sample_size']}")
        return int(min_ss)
