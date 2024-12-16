# textract
```bash
brew cask install xquartz
brew install poppler antiword unrtf tesseract swig
pip install textract
```

I had to use this because the above failed:
`pip install textract==1.6.3`

I also did this
`pip install chardet pdfminer.six docx2txt extract-msg==0.28.7 SpeechRecognition xlrd EbookLib`

and 

`pip install pypdf`

get the BioBERT-mnli-snli-scinli-scitail-mednli-stsb folder in root