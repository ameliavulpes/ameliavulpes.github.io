import math


def generate_svg_boxes(
    filename="numbered_circle_boxes.svg",
    num_boxes=8,
    viewbox_size=800,
    circle_radius=180,  # Made smaller
    box_size=60,  # Square dimensions
    stroke_color="#FFFFFF",
    stroke_width=3,
    font_size=28,
):
    center = viewbox_size / 2
    half_box = box_size / 2

    svg_elements = []

    for i in range(num_boxes):
        # Calculate clockwise angle (0 at the top)
        angle_rad = 2 * math.pi * i / num_boxes
        angle_deg = math.degrees(angle_rad)

        # Center position of each box along the circle circumference
        cx = center + circle_radius * math.sin(angle_rad)
        cy = center - circle_radius * math.cos(angle_rad)

        # Create rotated group for each box and text element
        group = f"""  <g transform="translate({cx:.2f}, {cy:.2f}) rotate({angle_deg:.2f})">
    <rect x="{-half_box}" y="{-half_box}" width="{box_size}" height="{box_size}" 
          fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}" />
    <text x="0" y="0" fill="{stroke_color}" font-size="{font_size}" font-weight="bold" 
          font-family="sans-serif" dominant-baseline="central" text-anchor="middle">
      {i}
    </text>
  </g>"""
        svg_elements.append(group)

    # Assemble complete SVG
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {viewbox_size} {viewbox_size}" width="100%" height="100%">
{chr(10).join(svg_elements)}
</svg>"""

    with open(filename, "w") as f:
        f.write(svg_content)

    return svg_content


if __name__ == "__main__":
    generate_svg_boxes()
