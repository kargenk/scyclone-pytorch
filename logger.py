from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """lossなどのスカラー値のログを描画する"""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def image_summary(self, tag, image, step):
        """画像を描画する."""
        self.writer.add_image(tag, image, step)
        self.writer.flush()

    def figure_summary(self, tag, figure, step):
        """matplotlibの画像を描画する，混同行列用"""
        self.writer.add_figure(tag, figure, step)
        self.writer.flush()

    def image_batch_summary(self, tag, images, step, format='NCHW'):
        """白黒画像としてバッチ画像を描画する，log_melsp用"""
        self.writer.add_images(tag, images, step, dataformats=format)
        self.writer.flush()

    def audio_summary(self, tag, audio, step, sr):
        """音声を埋め込む"""
        self.writer.add_audio(tag, audio, step, sr)
        self.writer.flush()

    def model_summary(self, model, input_to_model):
        """モデル構造を埋め込む"""
        self.writer.add_graph(model, input_to_model)
        self.writer.flush()
