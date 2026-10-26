"""Replace panel contents while retaining the author's six-panel layout."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

path = Path(__file__).with_name('nilm_related_work_concept.drawio')
tree = ET.parse(path)
root = tree.find('.//root')
captions = ['7', '16', '25', '45', '64', '96']
titles = ['(a) Parallel task heads', '(b) State-conditioned power',
          '(c) Task-specific expert routing', '(d) Shared vector outputs',
          '(e) Appliance-specific branches', '(f) Cross-appliance interaction']
origins = []
for cid, title in zip(captions, titles):
    cell = root.find(f"mxCell[@id='{cid}']")
    g = cell.find('mxGeometry')
    origins.append(float(g.get('x')))
    cell.set('value', title)
keep = {'0', '1', '3', '4', '5', '6', '43', *captions}
for cell in list(root):
    if cell.get('id') not in keep:
        root.remove(cell)

palette = {
    'input': ('#F7F8F9', '#7E8892'), 'shared': ('#E8EDF1', '#667B8C'),
    'power': ('#DBEAF3', '#507C96'), 'state': ('#F4E5D2', '#AB7C43'),
    'private': ('#E6EEE8', '#66856E'), 'relation': ('#EAE3F0', '#837095'),
}
counter = 0
positions = {}
def box(value, x, y, w, h, kind='shared', size=16, extra=''):
    global counter
    counter += 1
    cid = f'new-{counter}'
    fill, stroke = palette[kind]
    style = (f'rounded=0;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;'
             f'spacing=2;strokeWidth=1.4;fillColor={fill};strokeColor={stroke};'
             f'fontFamily=Arial;fontSize={size};fontColor=#27333D;{extra}')
    cell = ET.SubElement(root, 'mxCell', id=cid, value=value, style=style,
                         vertex='1', parent='1')
    ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w),
                  height=str(h), **{'as': 'geometry'})
    positions[cid] = (x, y, w, h)
    return cid

def note(value, x, y, w, h=20, size=13):
    return box(value, x, y, w, h, size=size,
               extra='fillColor=none;strokeColor=none;')

def link(source, target, points=(), start=(.5, 1), end=(.5, 0),
         purple=False, dashed=False):
    global counter
    counter += 1
    style = ('edgeStyle=none;html=1;rounded=0;endArrow=block;endFill=1;'
             'endSize=6;strokeWidth=1.4;jumpStyle=arc;jumpSize=7;'
             f'strokeColor={"#837095" if purple else "#45525D"};'
             f'dashed={int(dashed)};exitX={start[0]};exitY={start[1]};'
             f'entryX={end[0]};entryY={end[1]};exitPerimeter=0;entryPerimeter=0;')
    cell = ET.SubElement(root, 'mxCell', id=f'new-{counter}', source=source,
                         target=target, parent='1', edge='1', style=style)
    g = ET.SubElement(cell, 'mxGeometry', relative='1', **{'as':'geometry'})
    if points:
        arr = ET.SubElement(g, 'Array', **{'as':'points'})
        for x, y in points:
            ET.SubElement(arr, 'mxPoint', x=str(x), y=str(y))

def split(source, targets, y):
    x, _, w, _ = positions[source]
    for target in targets:
        tx, _, tw, _ = positions[target]
        link(source, target, [(x+w/2, y), (tx+tw/2, y)])

# Row 1: one appliance, power/state objective interaction.
x = origins[0]
h = box('Shared feature h<sub>i</sub>', x+75, 75, 240, 35)
p = box('Power head', x+35, 145, 135, 38, 'power')
s = box('State head', x+220, 145, 135, 38, 'state')
split(h, [p,s], 127)
po = box('Power y&#770;<sub>i</sub>', x+35, 230, 135, 32, 'power')
so = box('ON probability p<sub>i</sub>', x+220, 230, 135, 32, 'state', 15)
link(p,po); link(s,so)
note('Shared features; separate predictions', x+15, 274, 360)

x = origins[1]
h = box('Shared feature h<sub>i</sub>', x+75, 75, 240, 35)
p = box('Raw power head', x+25, 139, 145, 36, 'power')
s = box('State head', x+220, 139, 145, 36, 'state')
split(h,[p,s],124)
gate = box('Power gate', x+25, 202, 145, 33, 'relation')
po = box('Power y&#770;<sub>i</sub>', x+25, 263, 145, 30, 'power')
so = box('ON probability p<sub>i</sub>', x+220, 263, 145, 30, 'state', 15)
link(p,gate); link(gate,po); link(s,so)
link(s,gate,[(x+292.5,218.5)],end=(1,.5))
note('p<sub>i</sub>', x+223,181,35,18,14)

x = origins[2]
h = box('Input / shared features', x+75,75,240,32)
experts = [box(f'Expert {j+1}', x+40+j*110,132,90,30,size=14) for j in range(3)]
split(h,experts,119)
p = box('Power routing',x+45,206,130,32,'relation',15)
s = box('State routing',x+215,206,130,32,'relation',15)
for j, expert in enumerate(experts):
    link(expert,p,end=((j+1)/4,0))
    link(expert,s,end=((j+1)/4,0))
po=box('Power tower',x+45,264,130,30,'power',15)
so=box('State tower',x+215,264,130,30,'state',15)
link(p,po); link(s,so)

# Row 2: common input and paired outputs make the body comparison explicit.
x = origins[3]
inp=box('Aggregate input',x+65,377,260,34,'input')
enc=box('Shared encoder / decoder',x+55,433,280,40)
link(inp,enc)
p=box('Power vector head',x+25,521,150,36,'power',15)
s=box('State vector head',x+215,521,150,36,'state',15)
split(enc,[p,s],497)
po=box('y&#770;<sub>1</sub>, ... , y&#770;<sub>M</sub>',x+25,605,150,32,'power')
so=box('p<sub>1</sub>, ... , p<sub>M</sub>',x+215,605,150,32,'state')
link(p,po); link(s,so)

for index, interaction in [(4,False),(5,True)]:
    x=origins[index]
    inp=box('Aggregate input',x+65,377,260,34,'input')
    enc=box('Shared encoder',x+65,433,260,40)
    link(inp,enc)
    dec=[]
    for offset, label in [(25,'1'),(235,'M')]:
        dec.append(box(f'Decoder {label}',x+offset,507,130,36,'private'))
    split(enc,dec,490)
    note('...',x+175,514,40,22,21)
    heads=[]
    for offset,label in [(25,'1'),(235,'M')]:
        heads.append(box(f'Paired head {label}',x+offset,605,130,32,'private',15))
    if interaction:
        refine=[]
        for offset in [25,235]:
            refine.append(box('Refine',x+offset,568,130,25,'relation',14))
        for d,r,h in zip(dec,refine,heads):
            link(d,r); link(r,h)
        link(dec[0],refine[1],start=(1,1),end=(0,0),purple=True)
        link(dec[1],refine[0],start=(0,1),end=(1,0),purple=True)
    else:
        for d,h in zip(dec,heads):
            link(d,h)
        note('Independent appliance streams',x+30,560,330,23,14)

# Compact labels explain the nesting without duplicating every output head.
note('Paired head: power + state; see (a)-(c). Purple arrows: cross-appliance messages.',
     55,692,1140,24,14)
ids={c.get('id') for c in root}
for c in root:
    for attribute in ['source','target']:
        if c.get(attribute):
            assert c.get(attribute) in ids
backup=Path('tmp/pdfs/related_work/nilm_related_work_before_head_body.drawio')
backup.parent.mkdir(parents=True,exist_ok=True)
if not backup.exists():
    shutil.copy2(path,backup)
ET.indent(tree,space='  ')
tree.write(path,encoding='utf-8',xml_declaration=False)
print(f'Updated {path}: {len(ids)} editable cells; panel origins and dividers preserved.')
