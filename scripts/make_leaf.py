"""Generate the project's original geometric leaf target; no external image source."""

from pathlib import Path
from PIL import Image, ImageDraw

if __name__ == '__main__':
    output = Path(__file__).resolve().parents[1] / 'assets/targets/leaf.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGBA', (32, 32))
    draw = ImageDraw.Draw(image)
    draw.polygon([(16, 2), (25, 9), (28, 17), (23, 24), (16, 28),
                  (9, 24), (4, 17), (7, 9)], fill=(45, 170, 75, 255))
    draw.line([(16, 6), (16, 30)], fill=(20, 95, 40, 255), width=2)
    image.save(output)
