import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf


class Tensorboard:
    """
    tensorboard = Tensorboard('logs')
    x = np.arange(1,101)
    y = 20 + 3 * x + np.random.random(100) * 100

    # Log simple values
    for i in range(0,100):
        tensorboard.log_scalar('value', y[i], i)

    # Log plots
    fig = plt.figure()
    plt.plot(x, y, 'o')
    plt.close()
    tensorboard.log_plot('example_plot', fig, 0)

    # Log histograms
    rng = np.random.RandomState(10)
    a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
    tensorboard.log_histogram('example_hist', a, 0, 'auto')

    tensorboard.close()

    info = {'loss': loss.item(), 'accuracy': accuracy.item()}

    for tag, value in info.items():
        tensorboard.scalar_summary(tag, value, step + 1)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        tensorboard.histo_summary(tag, value.data.cpu().numpy(), step + 1)
        tensorboard.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

    """

    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_histogram(self, tag, values, global_step, bins):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_plot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                       height=img_ar.shape[0],
                                       width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()


def mkdirs(name):
    # Try to make the logs folder but return error if already exist to avoid overwriting a model
    log_path = os.path.join('logs/', name)
    save_path = os.path.join('trained_models', name)
    try:
        os.mkdir(log_path)
        os.mkdir(save_path)
    except FileExistsError:
        pass
        # raise ValueError('This name is already taken !')
    save_name = os.path.join(save_path, name + 'pth')
    return log_path, save_name
