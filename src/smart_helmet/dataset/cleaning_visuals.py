"""Static review evidence: copies only, no model predictions."""

from html import escape
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw

from .visual_review import draw_annotation

FOLDERS = {'P0': 'P0_cross_split', 'P1': 'P1_tiny_bbox',
           'P2': 'P2_unusual_resolution', 'P3': 'P3_source_variants'}


def panel(image, title, width=560):
    """Preserve the whole image; resize only the generated display copy."""
    lines = []
    for line in title.splitlines():
        lines.extend(textwrap.wrap(line, width=85) or [''])
    top = len(lines) * 15 + 16
    copy = image.copy()
    copy.thumbnail((width, 660))
    canvas = Image.new('RGB', (width, top + copy.height + 12), '#f4f6f8')
    ImageDraw.Draw(canvas).multiline_text((8, 8), '\n'.join(lines), fill='#18232e', spacing=5)
    canvas.paste(copy, ((width - copy.width)//2, top))
    return canvas


def join_panels(panels):
    width = sum(p.width for p in panels)
    result = Image.new('RGB', (width, max(p.height for p in panels)), '#f4f6f8')
    x = 0
    for item in panels:
        result.paste(item, (x, 0))
        x += item.width
    return result


def generate_comparison(row, records, by_image, root, output):
    priority = row['priority']
    members = row['members']
    panels = []
    for index, name in enumerate(members):
        record = records[name]
        canvas = draw_annotation(root / name, by_image[name])
        title = f"{row['review_id']} | {'A' if index == 0 else 'B' if index == 1 else index + 1} | {record['split'].upper()}\n{record['filename']}\n{record['image_width']}x{record['image_height']} | Ground Truth: 0 With Helmet (green), 1 Without Helmet (red)"
        if priority == 'P0':
            title += f"\npHash distance = {row['phash_distance']} | candidate, human review required"
        if priority == 'P1':
            target = next(b for b in by_image[name] if b['line_number'] == row['bbox_line_number'])
            w, h = canvas.size
            x, y, bw, bh = (target[k] for k in ('x','y','normalized_width','normalized_height'))
            edges = ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)
            ImageDraw.Draw(canvas).rectangle(edges, outline='#ffe600', width=3)
            title += f"\nTARGET: {row['class_id']} {row['class_name']} | line {row['bbox_line_number']} | normalized area = {row['normalized_area']:.8g}"
            panels.append(panel(canvas, title))
            pad_x, pad_y = max(25, bw*w*2), max(25, bh*h*2)
            crop_bounds = (max(0,int(edges[0]-pad_x)), max(0,int(edges[1]-pad_y)),
                           min(w,int(edges[2]+pad_x+1)), min(h,int(edges[3]+pad_y+1)))
            # Crop raw pixels so enlarged labels cannot obscure the tiny head.
            with Image.open(root / name) as original:
                crop = original.convert('RGB').crop(crop_bounds)
            original_crop_width, original_crop_height = crop.size
            scale = min(500/crop.width, 500/crop.height)
            crop = crop.resize((max(1,round(crop.width*scale)), max(1,round(crop.height*scale))), Image.Resampling.NEAREST)
            sx, sy = crop.width / original_crop_width, crop.height / original_crop_height
            ImageDraw.Draw(crop).rectangle(((edges[0]-crop_bounds[0])*sx,(edges[1]-crop_bounds[1])*sy,
                                            (edges[2]-crop_bounds[0])*sx,(edges[3]-crop_bounds[1])*sy),
                                           outline='#ffe600',width=2)
            panels.append(panel(crop, f'Context zoom | yellow = target bbox\n{row["class_name"]} | normalized area = {row["normalized_area"]:.8g}\nEnlarged display only; original remains unchanged.'))
            continue
        if priority == 'P3':
            title += f"\nSource group: {row['source_group_id']}\nObjects: {record['number_of_objects']} | informational sample"
        panels.append(panel(canvas, title))
    target = output / 'review_images' / FOLDERS[priority] / (row['review_id'] + '.png')
    join_panels(panels).save(target)
    return target.relative_to(output).as_posix()


def render_html(queue, summary):
    counts = {p: sum(r['priority'] == p for r in queue) for p in FOLDERS}
    labels = {'P0':'CROSS-SPLIT', 'P1':'TINY BBOX', 'P2':'RESOLUTION', 'P3':'SOURCE VARIANT SAMPLE'}
    cards = '\n'.join(f'        <a class="card" href="#{p}"><strong>{counts[p]}</strong>{p} {labels[p]}</a>' for p in FOLDERS)
    sections = []
    for priority in FOLDERS:
        cases = []
        for row in queue:
            if row['priority'] != priority:
                continue
            members = '<br>\n            '.join(escape(name) for name in row['members'])
            src = '../' + row['visualization_path']
            cases.append(
                f'        <article id="{escape(row["review_id"])}">\n'
                f'          <h3>{escape(row["review_id"])}</h3>\n'
                f'          <p class="files">\n            {members}\n          </p>\n'
                f'          <p><b>Suggestion only:</b> {escape(row["suggested_action"])}.\n'
                f'            {escape(row["reason"])}</p>\n'
                f'          <p>{escape(row["evidence"])}</p>\n'
                f'          <a href="{escape(src, quote=True)}">\n'
                f'            <img loading="lazy" src="{escape(src, quote=True)}"\n'
                f'                 alt="{escape(row["review_id"])} ground-truth review">\n'
                f'          </a>\n'
                f'        </article>'
            )
        sections.append(
            f'      <section id="{priority}">\n'
            f'        <h2>{priority} {labels[priority]}</h2>\n'
            + '\n'.join(cases) + '\n      </section>'
        )
    body_sections = '\n'.join(sections)
    return f'''<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>DATASET CLEANING REVIEW</title>
    <style>
      body {{ font: 16px/1.5 system-ui,sans-serif; background: #edf1f5; color: #172536; margin: 0; }}
      main {{ max-width: 1250px; margin: auto; padding: 24px; }}
      h1, h2 {{ line-height: 1.2; }}
      .cards {{ display: flex; flex-wrap: wrap; gap: 12px; }}
      .card, article {{ background: white; border: 1px solid #ccd5df; border-radius: 8px; padding: 18px; }}
      .card {{ flex: 1; min-width: 180px; color: inherit; text-decoration: none; }}
      .card strong {{ display: block; font-size: 32px; }}
      article {{ margin: 18px 0; }}
      img {{ max-width: 100%; height: auto; display: block; }}
      .files {{ overflow-wrap: anywhere; font: 13px monospace; }}
      .notice {{ border-left: 5px solid #c67c00; padding: 12px; background: #fff4da; }}
      section {{ scroll-margin-top: 12px; }}
      a {{ color: #155ca5; }}
    </style>
  </head>
  <body>
    <main>
      <h1>DATASET CLEANING REVIEW</h1>
      <p class="notice">
        Review preparation only. No dataset changes have been applied. P0 is blocking.
        pHash similarity is not proof of identical pixels. Click any comparison to inspect the full image.
      </p>
      <p>
        <a href="../review_decisions.csv">Decision CSV</a> |
        <a href="../review_queue.csv">Review queue</a> |
        <a href="../cleaning_policy.md">Cleaning policy</a>
      </p>
      <p>
        Use scripts/review_cleaning_cases.py for interactive P0 decisions, then run the validator.
        This static report does not edit or apply decisions. P3 is informational for same-split groups.
      </p>
      <div class="cards">
{cards}
      </div>
{body_sections}
    </main>
  </body>
</html>
'''
