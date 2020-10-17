import stanza

# if the resource has not been downloaded
# stanza.download('en')
# stanza_nlp = stanza.Pipeline('en')

list_image_name = open('../data/task1/image_splits/all_images.txt','r').read().split('\n')
list_caption = open('../data/task1/raw/en_labels/all_label_en.txt','r').read().split('\n') 

total_images = len(list_image_name)

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

for i in range(total_images):
  image_name = list_image_name[i]
  caption = list_caption[i]
  doc = nlp(caption)
  
  print(f'{i}: {image_name}')
  
  target_file =  open(f'en/{image_name[:-4]}.txt', 'w')
  target_file.write(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
  target_file.close()
