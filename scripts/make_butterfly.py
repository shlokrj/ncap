"""Generate an original geometric butterfly target; no external image source."""

from pathlib import Path
from PIL import Image, ImageDraw

if __name__ == '__main__':
    output = Path(__file__).resolve().parents[1] / 'assets/targets/butterfly.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGBA', (32, 32))
    draw = ImageDraw.Draw(image)
    for mirror in (False, True):
        def points(vertices):
            return [(31 - x if mirror else x, y) for x, y in vertices]

        draw.polygon(points([(14, 13), (7, 3), (2, 4), (3, 13),
                             (8, 18), (14, 19)]), fill=(235, 135, 35, 255))
        draw.polygon(points([(14, 17), (6, 18), (4, 24), (8, 28),
                             (13, 25), (15, 21)]), fill=(205, 75, 40, 255))
        draw.line(points([(15, 10), (12, 5)]), fill=(55, 40, 35, 255))
    draw.rectangle((14, 10, 17, 24), fill=(55, 40, 35, 255))
    image.save(output)
