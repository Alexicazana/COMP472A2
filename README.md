# COMP472-A2
**Alexandra Zana** 

***ID: 40131077***

**Brayden Tsitsirides**

------------------

## Notes on running the program
Assuming you have Python 3.11 and Genism downloaded.

- There seems to be an issue with the Gensim library's loader; I posted a StackOverflow question about it here:
https://stackoverflow.com/questions/77567868/jsondecodeerror-and-filenotfounderror-when-loading-word2vec-google-news-300-mode

- Therefore, I downloaded the model manually from Google's drive for POC:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g

- This alternative method works correctly, but it is a large file (around 1.53 GB), so it may take a little longer to download.

- EDIT: Solved Gensim library loader issue, will look into a permanent fix for the library over the break
- Refer to StackOverflow article below related to model run time optimization: https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time?rq=2


--------
## Task 1
- Initial model: Google News Word2Vec model
- Analysis in the analysis.csv file:
![Analysis](Task1/task1_analysis_output.png)



**Test case 1:**
Modifying the synonym.csv file so that none of the words are in the model's vocabulary. The model should not be able to find any synonyms for any of the words in the synonym.csv file, so it should return 'guess' for that word inside the generated word2vec .csv file:
Guessword test case:
![Guessword testcase1](guessword_testcase_task1.png)
Guessword analysis:
![Guessword analysis](guessword_testcase1_analysis.png)
Guessword modification output:
![Guessword output](guessword_testcase1_analysis.png)



MISC
https://stackoverflow.com/questions/39549248/how-to-load-a-pre-trained-word2vec-model-file-and-reuse-it
