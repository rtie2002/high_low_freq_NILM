"""Build the editable six-panel related-work figure using native draw.io cells.

Export the resulting .drawio with draw.io Desktop; the PDF in the manuscript
must be exported from this file, rather than from a separately drawn preview.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


OUTPUT = Path(__file__).with_name("nilm_related_work_concept.drawio")
COLORS = {
    "shared": ("#E8EDF1", "#667B8C"),
    "private": ("#E6EEE8", "#66856E"),
    "power": ("#DBEAF3", "#507C96"),
    "state": ("#F4E5D2", "#AB7C43"),
    "relation": ("#EAE3F0", "#837095"),
    "input": ("#F7F8F9", "#7E8892"),
}


class Diagram:
    def __init__(self):
        self.file = ET.Element("mxfile", host="Electron", compressed="false")
        page = ET.SubElement(self.file, "diagram", id="nilm-related-work", name="Related work")
        model = ET.SubElement(
            page, "mxGraphModel", dx="1240", dy="800", grid="1", gridSize="5",
            guides="1", tooltips="1", connect="1", arrows="1", fold="1",
            page="0", pageScale="1", pageWidth="1240", pageHeight="800",
            background="#FFFFFF", math="0", shadow="0",
        )
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.count = 1

    def cell(self, value, x, y, width, height, style):
        self.count += 1
        item = ET.SubElement(
            self.root, "mxCell", id=str(self.count), value=value, style=style,
            vertex="1", parent="1",
        )
        ET.SubElement(item, "mxGeometry", x=str(x), y=str(y), width=str(width),
                      height=str(height), **{"as": "geometry"})
        return str(self.count)

    def text(self, value, x, y, width, height=27, size=18, bold=False):
        return self.cell(value, x, y, width, height,
                         "text;html=1;strokeColor=none;fillColor=none;"
                         "whiteSpace=wrap;overflow=hidden;align=center;verticalAlign=middle;"
                         f"fontFamily=Arial;fontSize={size};fontColor=#27333D;"
                         f"fontStyle={1 if bold else 0};")

    def box(self, value, x, y, width, height=40, kind="shared", size=18):
        fill, stroke = COLORS[kind]
        return self.cell(value, x, y, width, height,
                         "rounded=0;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"
                         "spacing=3;strokeWidth=1.4;"
                         f"fillColor={fill};strokeColor={stroke};"
                         f"fontFamily=Arial;fontSize={size};fontColor=#27333D;")

    def edge(self, source, target, points=(), dashed=False,
             exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0):
        self.count += 1
        item = ET.SubElement(
            self.root, "mxCell", id=str(self.count), edge="1", parent="1",
            source=source, target=target,
            style="edgeStyle=none;rounded=0;html=1;endArrow=block;endFill=1;"
                  "endSize=7;strokeWidth=1.5;strokeColor=#45525D;"
                  f"dashed={1 if dashed else 0};"
                  f"exitX={exit_x};exitY={exit_y};exitPerimeter=0;"
                  f"entryX={entry_x};entryY={entry_y};entryPerimeter=0;",
        )
        geometry = ET.SubElement(item, "mxGeometry", relative="1", **{"as": "geometry"})
        if points:
            array = ET.SubElement(geometry, "Array", **{"as": "points"})
            for x, y in points:
                ET.SubElement(array, "mxPoint", x=str(x), y=str(y))

    def rule(self, x, y, width, height):
        self.cell("", x, y, width, height,
                  "shape=line;strokeColor=#D7DDE2;strokeWidth=1;endArrow=none;")

    def pair(self, center, y, appliance):
        power = self.box(f"Power {appliance}", center - 77, y, 73, 35, "power", 16)
        state = self.box(f"State {appliance}", center + 4, y, 73, 35, "state", 16)
        return power, state

    def branch_pair(self, source, center, y, appliance):
        power, state = self.pair(center, y, appliance)
        self.edge(source, power, [(center, y - 12), (center - 40.5, y - 12)])
        self.edge(source, state, [(center, y - 12), (center + 40.5, y - 12)])

    def save(self):
        ET.indent(self.file, space="  ")
        ET.ElementTree(self.file).write(OUTPUT, encoding="utf-8", xml_declaration=True)
        ids = {cell.attrib["id"] for cell in self.root}
        for cell in self.root:
            if cell.get("edge"):
                assert cell.get("source") in ids and cell.get("target") in ids
        print(f"Saved {OUTPUT.name}: {len(ids)} editable cells")


def main():
    d = Diagram()
    d.text("Prediction objectives", 10, 0, 1220, 30, 22, True)
    for x in (412, 828):
        d.rule(x, 42, 0, 303)
        d.rule(x, 398, 0, 342)

    # (a): A directly learned multi-target regressor; states are derived.
    d.text("(a) Multi-target power regression", 12, 37, 390, 28, 19, True)
    d.text("Conv-NILM-Net", 12, 66, 390, 23, 16)
    a0 = d.box("Aggregate input", 78, 100, 262, 35, "input")
    a1 = d.box("Shared feature extraction", 78, 155, 262)
    a2 = d.box("Multi-appliance power outputs", 78, 215, 262, 38, "power")
    a3 = d.box("Optional: thresholded states", 78, 287, 262, 32, "state", 16)
    d.edge(a0, a1)
    d.edge(a1, a2)
    d.edge(a2, a3, dashed=True)

    # (b): The classifier may output labels directly or decode label sets.
    d.text("(b) Multi-label state classification", 424, 37, 390, 28, 19, True)
    d.text("Buddhahai et al.; Tanoni et al.", 424, 66, 390, 23, 16)
    b0 = d.box("Aggregate input / features", 494, 100, 262, 35, "input")
    b1 = d.box("Per-label / label-set classifier", 494, 155, 262, 40, "shared", 17)
    b2 = d.box("Multi-appliance state outputs", 494, 215, 262, 38, "state")
    b3 = d.box("Optional: state power lookup", 494, 287, 262, 32, "power", 16)
    d.edge(b0, b1)
    d.edge(b1, b2)
    d.edge(b2, b3, dashed=True)

    # (c): Explicitly supervised paired tasks, nested within appliance streams.
    d.text("(c) Joint multi-appliance estimation", 840, 37, 390, 28, 19, True)
    d.text("UNet-NILM; MATNilm", 840, 66, 390, 23, 16)
    c0 = d.box("Aggregate input", 910, 100, 262, 35, "input")
    c1 = d.box("Shared representation", 910, 155, 262)
    c2 = d.box("Appliance 1", 863, 218, 150, 38, "private")
    c3 = d.box("Appliance M", 1069, 218, 150, 38, "private")
    d.text("...", 1019, 220, 44, 30, 24)
    d.edge(c0, c1)
    d.edge(c1, c2, [(1041, 206), (938, 206)])
    d.edge(c1, c3, [(1041, 206), (1144, 206)])
    d.branch_pair(c2, 938, 286, "1")
    d.branch_pair(c3, 1144, 286, "M")

    d.rule(12, 351, 1218, 0)
    d.text("Where and how information is shared", 10, 361, 1220, 30, 22, True)

    # (d): The appliance-to-objective hierarchy, isolating its branching pattern.
    d.text("(d) Shared encoder, private branches", 12, 399, 390, 28, 19, True)
    d.text("Hierarchical branch pattern: MATNilm", 12, 428, 390, 23, 16)
    q0 = d.box("Aggregate input", 78, 464, 262, 34, "input")
    q1 = d.box("Shared encoder", 78, 520, 262, 40)
    q2 = d.box("Decoder 1", 32, 606, 151, 42, "private")
    q3 = d.box("Decoder M", 238, 606, 151, 42, "private")
    d.edge(q0, q1)
    d.edge(q1, q2, [(209, 582), (107.5, 582)])
    d.edge(q1, q3, [(209, 582), (313.5, 582)])
    d.text("...", 187, 610, 44, 30, 24)
    d.branch_pair(q2, 107.5, 698, "1")
    d.branch_pair(q3, 313.5, 698, "M")
    d.text("Appliance-specific paired objectives", 12, 747, 390, 23, 16)

    # (e): SAMNet's single-appliance task gates; do not label it multi-label.
    d.text("(e) Task-specific expert routing", 424, 399, 390, 28, 19, True)
    d.text("SAMNet: one appliance, two tasks", 424, 428, 390, 23, 16)
    e0 = d.box("Aggregate input", 494, 464, 262, 34, "input")
    expert_nodes = [d.box(f"Expert {k + 1}", x, 530, 92, 38)
                    for k, x in enumerate((467, 579, 691))]
    for node, x in zip(expert_nodes, (513, 625, 737)):
        d.edge(e0, node, [(625, 513), (x, 513)])
    g1 = d.box("Gate", 432, 585, 65, 32, "relation", 16)
    g2 = d.box("Gate", 753, 585, 65, 32, "relation", 16)
    d.edge(e0, g1, [(449, 481), (449, 574), (464.5, 574)], exit_x=0, exit_y=0.5)
    d.edge(e0, g2, [(801, 481), (801, 574), (785.5, 574)], exit_x=1, exit_y=0.5)
    r1 = d.box("State fusion", 506, 616, 111, 34, "relation", 16)
    r2 = d.box("Power fusion", 633, 616, 111, 34, "relation", 16)
    for node in expert_nodes:
        d.edge(node, r1)
        d.edge(node, r2)
    d.edge(g1, r1, [(464.5, 633)], dashed=True, entry_x=0, entry_y=0.5)
    d.edge(g2, r2, [(785.5, 633)], dashed=True, entry_x=1, entry_y=0.5)
    s = d.box("State tower", 506, 667, 111, 30, "state", 16)
    p = d.box("Power tower", 633, 667, 111, 30, "power", 16)
    so = d.box("State", 506, 716, 111, 27, "state", 16)
    po = d.box("Power", 633, 716, 111, 27, "power", 16)
    for source, target in ((r1, s), (r2, p), (s, so), (p, po)):
        d.edge(source, target)
    d.text("Multi-appliance routing: Liu and Fan (2026)", 424, 751, 390, 23, 15)

    # (f): Feature exchange after private decoding, as in MART-NILM.
    d.text("(f) Interaction among private streams", 840, 399, 390, 28, 19, True)
    d.text("Task-interaction principle; MART-NILM", 840, 428, 390, 23, 16)
    f0 = d.box("Aggregate input", 910, 464, 262, 34, "input")
    f1 = d.box("Shared encoder", 910, 520, 262, 40)
    f2 = d.box("Decoder 1", 864, 589, 151, 38, "private")
    f3 = d.box("Decoder M", 1070, 589, 151, 38, "private")
    d.text("...", 1019, 592, 44, 30, 24)
    f4 = d.box("Cross-appliance feature interaction", 864, 653, 357, 32, "relation", 17)
    d.edge(f0, f1)
    d.edge(f1, f2, [(1041, 574), (939.5, 574)])
    d.edge(f1, f3, [(1041, 574), (1145.5, 574)])
    d.edge(f2, f4, entry_x=75.5 / 357)
    d.edge(f3, f4, entry_x=281.5 / 357)
    for center, appliance, fraction in ((939.5, "1", 75.5 / 357), (1145.5, "M", 281.5 / 357)):
        power, state = d.pair(center, 714, appliance)
        d.edge(f4, power, [(center, 699), (center - 40.5, 699)], exit_x=fraction)
        d.edge(f4, state, [(center, 699), (center + 40.5, 699)], exit_x=fraction)
    d.text("Cross-domain precedent: PAD-Net", 840, 758, 390, 23, 16)

    d.text("Schematics isolate design choices; mechanisms can be combined.",
           10, 794, 1220, 25, 16)
    d.save()


if __name__ == "__main__":
    main()
