# Topic-Modelling-for-PornHub-s-Titles
Topic Modelling for PornHub's Videos to exploit any differences in words used for general videos, and for videos appreciated by the female audience, to see if the video title affects the target.

Number of Titles:
1) General Videos: 48009
2) Popular With Women: 40570

Steps:
1) Scraping Using Beautiful Soup and Requests to obtain:
   1) Video Title;
   2) Video Rating;
   3) Video Views;
   4) Video Length;
   
    For General videos (link: https://www.pornhub.com/video) and for videos
    popular with women (link: https://www.pornhub.com/popularwithwomen);


2) Detect title language to delete each not english word, and any number
    to avoid any misleading result in the analysis;


3) Write all titles in two CSV, one for General Videos and one for videos Popular with Women;


4) Perform Topic Modelling for each set of titles using Latent Dirichlet Allocation and GridSearchCV, 
and produce two file.html with all the results and the most used words.


Issues:
Before performing LDA You have to change some attributes, in particular:
1) Inside "sklearn.py" of the package "pyLDAvis" for the visualization of the
results, You have to change inside the function "_get_vocab" the method "get_feature_names"
with "get_feature_names_out";
2) Inside "_prepare.py" of the package "pyLDAvis", You have to change inside the
method "drop" associated with head(R), the number "1" with the write axis=1, so
You have to specify the parameter "axis" to avoid this error.

Inside the Repository You'll find, except this README:
1) Topic_Modelling_PH.py, where You can find the script;
2) title_videos.csv, where You can find the titles used for the analysis in data 18/08/2023;
3) title_videos_fem.csv, as the previous point but with titles of videos popular with women;
4) LDA_panel_female videos.html, where You can find the visualization of the results of
LDA performed for videos popular with women;
5) LDA_panel_general videos.html, where You can find the visualization of the results of
LDA performed for general videos.