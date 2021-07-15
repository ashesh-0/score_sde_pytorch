import torch


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""
    def __call__(self, img1, img2):
        # img.shape should be batch, sz,sz,3)
        assert len(img1.shape) == 4
        mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        return psnr.cpu()


class MSE:
    def __call__(self, img1, img2):
        # img.shape should be batch, sz,sz,3)
        assert len(img1.shape) == 4
        mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
        return mse.cpu()


class EvalMetrics:
    def __init__(self, message_prefix=''):
        self._metrics = []
        self._metrics_val = {}
        self._msg = message_prefix

        self.add(MSE())
        self.add(PSNR())

    def key(self, metric):
        return metric.__class__.__name__

    def add(self, metric):
        assert self.key(metric) not in self._metrics
        self._metrics.append(metric)
        self._metrics_val[self.key(metric)] = []

    def update(self, target, prediction):
        for metric in self._metrics:
            self._metrics_val[self.key(metric)].append(metric(target, prediction))

    def get(self):
        output = {}
        for i, metric in enumerate(self._metrics):
            k = self.key(metric)
            output[k] = torch.cat(self._metrics_val[k])
            print(f'[{self.__class__.__name__} {self._msg}] {k}:{torch.mean(output[k]):.2f}'
                  f'+-{torch.std(output[k]):.3f}')
        return output
