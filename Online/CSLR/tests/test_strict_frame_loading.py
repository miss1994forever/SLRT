import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.VideoLoader import load_batch_video, read_jpg


class StrictFrameLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sample = Path(self.tmp.name) / 'sentence_frames-512x512' / 'sample'
        self.sample.mkdir(parents=True)

    def test_valid_images(self):
        Image.new('RGB', (512, 512), (100, 120, 140)).save(self.sample / '000000.jpg')
        frames = read_jpg(self.tmp.name, 'csl', [0], 1, 'sample', strict_frame_loading=True)
        self.assertEqual(frames.shape, (1, 320, 320, 3))
        self.assertGreater(frames.mean(), 90)

    def test_missing_and_corrupt_images_raise_with_path(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt):
                if corrupt:
                    (self.sample / '000000.jpg').write_bytes(b'not an image')
                with self.assertRaisesRegex(RuntimeError, 'sample.*000000.jpg') as ctx:
                    read_jpg(self.tmp.name, 'csl', [0], 1, 'sample', strict_frame_loading=True)
                self.assertIsNotNone(ctx.exception.__cause__)

    def test_strict_option_reaches_reader_from_batch(self):
        for dataset, name in [('csl', 'sample'), ('csl_iso', 'sample[0:4]')]:
            with self.subTest(dataset=dataset):
                with self.assertRaisesRegex(RuntimeError, 'sample.*000000.jpg'):
                    load_batch_video(
                        self.tmp.name, [name], [4], [4], dataset, False,
                        num_output_frames=4, ori_video_files=['sample'],
                        strict_frame_loading=True,
                    )

    def test_legacy_default_remains_opt_in(self):
        frames = read_jpg(self.tmp.name, 'csl', [0], 1, 'sample')
        self.assertEqual(frames.shape, (1, 320, 320, 3))
        self.assertFalse(frames.any())


if __name__ == '__main__':
    unittest.main()
