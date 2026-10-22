"""Native draw.io schematics of intra-appliance heads and appliance sharing."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

path=Path(__file__).with_name('nilm_related_work_concept.drawio')
tree=ET.parse(path)
root=tree.find('.//root')
captions=['7','16','25','45','64','96']
titles=['(a) Shared-bottom multitask heads','(b) Subtask-gated power',
        '(c) Scale/context-aware fusion','(d) Shared encoder, private decoders',
        '(e) Shared/private expert routing','(f) Private-stream interaction']
origins=[]
for cid,title in zip(captions,titles):
    cell=root.find(f"mxCell[@id='{cid}']")
    origins.append(float(cell.find('mxGeometry').get('x')))
    cell.set('value',title)
    cell.set('style',cell.get('style').replace('fontSize=19;','fontSize=18;'))
keep={'0','1','3','4','5','6','43',*captions}
for cell in list(root):
    if cell.get('id') not in keep:root.remove(cell)
palette={'input':('#F6F7F8','#7E8892'),'shared':('#E4EBF0','#667B8C'),
         'power':('#DCEAF2','#507C96'),'state':('#F3E5D2','#A8804E'),
         'private':('#E5EDE7','#708B75'),'relation':('#EAE4F0','#837095')}
counter=0
positions={}
def box(value,x,y,w,h,kind='shared',size=16,extra=''):
    global counter
    counter+=1
    cid=f'nested-{counter}'
    fill,stroke=palette[kind]
    style=(f'rounded=0;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;'
           f'spacing=2;strokeWidth=1.3;fillColor={fill};strokeColor={stroke};'
           f'fontFamily=Arial;fontSize={size};fontColor=#27333D;{extra}')
    c=ET.SubElement(root,'mxCell',id=cid,value=value,style=style,vertex='1',parent='1')
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
    positions[cid]=(x,y,w,h)
    return cid
def note(value,x,y,w,h=20,size=13,extra=''):
    return box(value,x,y,w,h,size=size,extra='fillColor=none;strokeColor=none;'+extra)
def circle(value,x,y,diameter=22,kind='relation',size=17):
    return box(value,x,y,diameter,diameter,kind,size,'shape=ellipse;')
def link(source,target,points=(),start=(.5,0),end=(.5,1),purple=False,dashed=False,arrow=True):
    global counter
    counter+=1
    style=('edgeStyle=none;html=1;rounded=0;'
           f'endArrow={"block" if arrow else "none"};endFill=1;'
           'endSize=5;strokeWidth=1.3;jumpStyle=arc;jumpSize=6;'
           f'strokeColor={"#837095" if purple else "#45525D"};'
           f'dashed={int(dashed)};dashPattern=4 3;exitX={start[0]};exitY={start[1]};'
           f'entryX={end[0]};entryY={end[1]};exitPerimeter=0;entryPerimeter=0;')
    c=ET.SubElement(root,'mxCell',id=f'nested-{counter}',source=source,target=target,
                    parent='1',edge='1',style=style)
    g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    if points:
        arr=ET.SubElement(g,'Array',**{'as':'points'})
        for x,y in points:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
def split(source,targets,y):
    x,_,w,_=positions[source]
    for target in targets:
        tx,_,tw,_=positions[target]
        link(source,target,[(x+w/2,y),(tx+tw/2,y)])
def paired_head(x,y,label):
    box('',x,y,150,94,'input',extra='fillColor=none;strokeColor=#84909A;dashed=1;dashPattern=5 4;')
    note(f'Appliance {label} head',x+4,y+2,142,18,13,extra='fontStyle=1;')
    # A paired head uses the same left-to-right convention throughout:
    # power and state branches rise independently; the state probability
    # gates the raw power only at the compact product node.
    p=box('P',x+25,y+66,44,17,'power',14)
    s=box('S',x+99,y+66,44,17,'state',14)
    # Centre the product above the power branch, so the primary power path is
    # visually vertical.  The state-to-product link is the only lateral link.
    gate=circle('&#8857;',x+47,y+39,18,size=14)
    outp=note('y&#770;<sub>'+label+'</sub>',x+35,y+19,42,18,14)
    outs=note('p<sub>'+label+'</sub>',x+106,y+20,34,20,14)
    link(p,gate)
    # One elbow takes the state probability into the product without
    # crossing either output path.
    link(s,gate,[(x+121,y+49),(x+65,y+49)],end=(1,.5))
    link(gate,outp)
    link(s,outs)
    anchor=box('',x+74,y+90,2,2,extra='fillColor=none;strokeColor=none;')
    split(anchor,[p,s],y+85)
    return anchor

# Top row: three multitask NILM head patterns as grouped in SAMNet Fig. 1.
# They are literature schematics, not a decomposition of MART-NILM itself.
x=origins[0]
inp=box('Input',x+118,266,150,24,'input',14)
bottom=box('Shared bottom',x+85,214,216,38,'shared',16)
state=box('State tower',x+37,142,108,34,'state',14)
power=box('Power tower',x+241,142,108,34,'power',14)
state_out=box('State output',x+37,78,108,27,'state',14)
power_out=box('Power output',x+241,78,108,27,'power',14)
link(inp,bottom); split(bottom,[state,power],198); link(state,state_out); link(power,power_out)

x=origins[1]
inp=box('Input',x+118,266,150,24,'input',14)
bottom=box('Shared bottom',x+85,214,216,38,'shared',16)
state=box('State tower',x+37,142,108,34,'state',14)
power=box('Power tower',x+241,142,108,34,'power',14)
state_out=box('State output',x+37,78,108,27,'state',14)
gate=circle('&#8857;',x+274,99,22,size=16)
power_out=box('Gated power output',x+219,48,132,27,'power',13)
link(inp,bottom); split(bottom,[state,power],198); link(state,state_out); link(power,gate); link(gate,power_out)
# State probability modulates only the final power estimate (SGN convention).
link(state,gate,[(x+91,121),(x+296,121)],start=(.5,0),end=(.5,1))

x=origins[2]
inp=box('Input',x+118,266,150,24,'input',14)
bottom=box('Shared bottom',x+85,222,216,32,'shared',16)
# SCA keeps task-specific multi-scale paths and exchanges context during fusion.
state_blocks=[box(f'S{j+1}',x+24+j*54,165,43,24,'state',12) for j in range(3)]
power_blocks=[box(f'P{j+1}',x+212+j*54,165,43,24,'power',12) for j in range(3)]
split(bottom,state_blocks,199); split(bottom,power_blocks,199)
state_fuse=box('State fusion',x+28,120,135,26,'state',13)
power_fuse=box('Power fusion',x+210,120,135,26,'power',13)
for q in state_blocks: link(q,state_fuse)
for q in power_blocks: link(q,power_fuse)
state_out=box('State output',x+37,64,108,27,'state',14)
gate=circle('&#8857;',x+264,79,22,size=16)
power_out=box('Gated power output',x+219,28,132,27,'power',13)
link(inp,bottom); link(state_fuse,state_out); link(power_fuse,gate); link(gate,power_out)
# Context is exchanged only between the two fusion modules, avoiding a false
# all-to-all expert graph.
link(state_fuse,power_fuse,start=(1,.5),end=(0,.5),dashed=True,arrow=False)
link(state_fuse,gate,[(x+95,105),(x+275,105)],start=(.5,0),end=(.5,1))

# Lower row: where appliance information is shared in multi-appliance models.
note('Where appliance information is shared',420,341,380,25,18,extra='fontStyle=1;')
x=origins[3]
out=[box('Power 1',x+14,435,72,27,'power',12),box('State 1',x+94,435,72,27,'state',12),
     box('Power M',x+214,435,72,27,'power',12),box('State M',x+294,435,72,27,'state',12)]
note('...',x+177,435,25,25,19)
dec=[box('Decoder 1',x+15,515,151,32,'private',14),box('Decoder M',x+215,515,151,32,'private',14)]
link(dec[0],out[0],end=(.32,1)); link(dec[0],out[1],end=(.68,1)); link(dec[1],out[2],end=(.32,1)); link(dec[1],out[3],end=(.68,1))
enc=box('Shared encoder',x+50,568,280,34,'shared',15)
split(enc,dec,555)
inp=box('Aggregate input',x+100,620,180,24,'input',14);link(inp,enc)

x=origins[4]
out=[box('Power 1',x+14,435,72,27,'power',12),box('State 1',x+94,435,72,27,'state',12),
     box('Power M',x+214,435,72,27,'power',12),box('State M',x+294,435,72,27,'state',12)]
note('...',x+177,435,25,25,19)
f1=circle('&#931;',x+73,485,22,size=16); fm=circle('&#931;',x+273,485,22,size=16)
for fusion, pair in [(f1,out[:2]),(fm,out[2:])]:
    link(fusion,pair[0],end=(.32,1));link(fusion,pair[1],end=(.68,1))
e1=box('Private E<sub>1</sub>',x+15,539,94,28,'private',12)
es=box('Shared E<sub>s</sub>',x+136,539,94,28,'shared',12)
em=box('Private E<sub>M</sub>',x+257,539,94,28,'private',12)
link(e1,f1);link(es,f1,[(x+183,517)],end=(.7,1));link(es,fm,[(x+183,517)],end=(.3,1));link(em,fm)
g1=box('g<sub>1</sub>',x+15,487,35,20,'relation',12);gm=box('g<sub>M</sub>',x+335,487,38,20,'relation',12)
link(g1,f1,start=(1,.5),end=(0,.5),dashed=True);link(gm,fm,start=(0,.5),end=(1,.5),dashed=True)
rep=box('Shared representation',x+50,585,280,28,'shared',15)
split(rep,[e1,es,em],576)
inp=box('Aggregate input',x+100,620,180,24,'input',14);link(inp,rep)

x=origins[5]
out=[box('Power 1',x+14,420,72,27,'power',12),box('State 1',x+94,420,72,27,'state',12),
     box('Power M',x+214,420,72,27,'power',12),box('State M',x+294,420,72,27,'state',12)]
note('...',x+177,420,25,25,19)
ref=[box('Refine 1',x+15,476,151,26,'relation',14),box('Refine M',x+215,476,151,26,'relation',14)]
for r,pair in [(ref[0],out[:2]),(ref[1],out[2:])]: link(r,pair[0],end=(.32,1));link(r,pair[1],end=(.68,1))
exchange=box('Cross-appliance feature interaction',x+15,512,351,24,'relation',12)
link(exchange,ref[0],start=(.25,0),purple=True);link(exchange,ref[1],start=(.75,0),purple=True)
dec=[box('Decoder 1',x+15,553,151,30,'private',14),box('Decoder M',x+215,553,151,30,'private',14)]
link(dec[0],exchange,end=(.25,1),purple=True);link(dec[1],exchange,end=(.75,1),purple=True)
enc=box('Shared encoder',x+50,593,280,28,'shared',15)
split(enc,dec,587)
inp=box('Aggregate input',x+100,635,180,23,'input',14);link(inp,enc)

note('Top: multitask head patterns in GRU/MGRU, SGN, and SCA.  Bottom: multi-appliance sharing patterns.',70,698,1130,20,13)
note('&#8857;: state-conditioned power.  &#931;: appliance-specific expert fusion.  Dashed arrows: routing weights.',70,720,1130,20,13)
ids={c.get('id') for c in root}
assert len(ids)==len(list(root)), 'Duplicate cell identifier'
for c in root:
    for a in ['source','target']:
        if c.get(a):assert c.get(a) in ids
backup=Path('tmp/pdfs/related_work/nilm_related_work_before_nested_heads.drawio')
backup.parent.mkdir(parents=True,exist_ok=True)
if not backup.exists():shutil.copy2(path,backup)
ET.indent(tree,space='  ')
tree.write(path,encoding='utf-8',xml_declaration=False)
print(f'Updated {path}: {len(ids)} native cells; author panel grid retained.')
