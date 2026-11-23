"""PDF technical QA plus page renders; review visual contact sheets separately."""
from pathlib import Path
import json,re,statistics,unicodedata
import pymupdf
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
P=Path(__file__).resolve().parents[1];pdf=P.parent/'dissertation_initial_draft.pdf'
doc=pymupdf.open(pdf);r=PdfReader(str(pdf));qa=P/'audit/rendered';qa.mkdir(exist_ok=True)
texts=[unicodedata.normalize('NFKC',p.get_text(flags=pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE)) for p in doc];alltext='\n'.join(texts)
(P/'audit/dissertation_extracted.txt').write_text(alltext)
fonts={};boxes=[];bad_sizes=[];body_gaps=[]
for i,page in enumerate(doc):
 if abs(page.rect.width-595.276)>.2 or abs(page.rect.height-841.89)>.2:bad_sizes.append(i+1)
 for f in page.get_fonts(full=True):
  xref=f[0]
  if xref not in fonts:
   name,ext,typ,data=doc.extract_font(xref)
   fonts[xref]={'name':name,'extension':ext,'type':typ,'embedded_bytes':len(data)}
 for b in page.get_text('dict')['blocks']:
  if 'lines' not in b:continue
  lines=[]
  for line in b['lines']:
   spans=line['spans']
   for s in spans:
    x0,y0,x1,y1=s['bbox']
    if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:boxes.append({'page':i+1,'bbox':s['bbox'],'text':s['text']})
   if spans and all(11.8<s['size']<12.1 for s in spans) and len(''.join(s['text'] for s in spans))>40:lines.append(line['spans'][0]['origin'][1])
  for a,b in zip(lines,lines[1:]):
   if 15<b-a<24:body_gaps.append(b-a)
 page.get_pixmap(matrix=pymupdf.Matrix(1.1,1.1),alpha=False).save(qa/f'page-{i+1:03d}.png')
# Contact sheets contain every rendered page, with visible page labels.
for start in range(0,len(doc),12):
 sheet=Image.new('RGB',(840,1212),'#dddddd');draw=ImageDraw.Draw(sheet)
 for j in range(start,min(start+12,len(doc))):
  im=Image.open(qa/f'page-{j+1:03d}.png');im.thumbnail((264,277))
  x=8+((j-start)%3)*280;y=8+((j-start)//3)*303
  sheet.paste(im,(x,y+18));draw.text((x,y),f'PDF page {j+1}',fill='black')
 sheet.save(qa/f'contact-{start//12+1:02d}.png')
# Abstract is extracted between the actual abstract heading and its following page.
ai=next(i for i,t in enumerate(texts) if t.lstrip().startswith('Abstract'))
at=texts[ai].strip();at=re.sub(r'^Abstract\s*','',at);at=re.sub(r'\s+[ivxlcdm]+\s*$','',at)
abstract_words=len(re.findall(r"\b[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*\b",at))
wc=json.loads((P/'audit/word_counts.json').read_text());assert abstract_words==wc['abstract_words'],(abstract_words,wc)
# Required order, using actual leading headings rather than mentions in contents.
ordered=[]
for i,t in enumerate(texts):
 first=' '.join(t.strip().splitlines()[:4])
 if i<3 or t.lstrip().startswith(('Abstract','Acknowledgements','Contents','List of Figures','List of Tables','List of Abbreviations','References','Chapter ','Appendix ')):
  ordered.append({'pdf_page':i+1,'leading_text':first[:160]})
log=(P/'build/main.log').read_text(errors='replace')
citation_links=[l for page in doc for l in page.get_links() if l.get('kind')==pymupdf.LINK_GOTO or l.get('nameddest')]
invalid_links=[l for l in citation_links if not 0<=l.get('page',-1)<len(doc)]
issues={'undefined_references':bool(re.search(r'(Citation .* undefined|Reference .* undefined|There were undefined)',log)), 'overfull_boxes':re.findall(r'Overfull \\[hv]box[^\n]*',log)}
result={'pdf':str(pdf),'pages':len(doc),'pdf_header':pdf.read_bytes()[:8].decode('ascii'),'all_pages_A4':not bad_sizes,'bad_page_sizes':bad_sizes,'pdf_version_at_least_1_4':tuple(map(int,pdf.read_bytes()[5:8].decode().split('.'))) >= (1,4),'internal_goto_links':len(citation_links),'citation_named_destinations':len({l['nameddest'] for l in citation_links if l.get('nameddest','').startswith('cite.')}),'invalid_internal_goto_links':len(invalid_links),'encrypted':r.is_encrypted,'encryption_dictionary_present':'/Encrypt' in r.trailer,'all_pages_searchable':all(len(t.strip())>10 for t in texts),'replacement_characters_in_text':alltext.count('\ufffd'),'all_fonts_embedded':all(v['embedded_bytes']>0 for v in fonts.values()),'font_programs':list(fonts.values()),'out_of_page_text':boxes,'body_baseline_median_pt':statistics.median(body_gaps),'body_font_pt':12,'body_baseline_to_font_ratio':statistics.median(body_gaps)/12,'body_spacing_method':'setspace onehalfspacing; physical body line pitch about18pt for12pt text','abstract_words':abstract_words,'pdf_whole_document_lexical_words':len(re.findall(r"\b[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*\b",alltext)),'source_chapter_prose_estimate':wc['chapter_total_approximation'],'section_page_order':ordered,'log_checks':issues,'visual_review_status':'PENDING_HUMAN_OR_AGENT_INSPECTION_OF_CONTACT_SHEETS','scope':'technical checks; not school approval, plagiarism check or proof certification'}
result['machine_checks_pass']=all([result['all_pages_A4'],result['pdf_version_at_least_1_4'],not invalid_links,len(citation_links)>0,result['replacement_characters_in_text']==0,not r.is_encrypted,not result['encryption_dictionary_present'],result['all_pages_searchable'],result['all_fonts_embedded'],not boxes,200<=abstract_words<=500,not issues['undefined_references'],17.5<result['body_baseline_median_pt']<18.5])
(P/'audit/PDF_VERIFICATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['font_programs','section_page_order']},indent=2))
print('Headings',ordered)
