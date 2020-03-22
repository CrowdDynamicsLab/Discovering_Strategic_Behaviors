## Dynamic Dual Attention Networks

This is a Multi-GPU PyTorch implementation of the Dynamic Dual Attention Networks used in our paper.

### Input

We provide the DBLP dataset for Year 2018 as a sample input. You can download it <a href="https://drive.google.com/drive/folders/1xdZLIsKnfxUFJHyae1wBZKgjLFJuc9Y_?usp=sharing">here</a>.

There are three files in the ``` cite_input``` folder:

```c_cite_inputs_2018.pkl```: <br />  
- c_active - the list of content_id of contents produced in 2018;
- c_position - the dictionary of (content_id, the index of content_id in c_active); 
- ca_adj - the array of \[content_id, author_id\] sorted in the order of c_active where author_id is an author of content_id;
- c_emb - the array of content embeddings sorted in the order of c_active;
- c_edgellh - the list of the log likelihoods of forming content-content citation links sorted in the order of c_active.

```a_cite_inputs_2018.pkl```: <br /> 
- a_active - the list of author_id of authors who produced contents in 2018;
- a_position - the dictionary of (author_id, the index of author_id in a_active); 
- ac_adj - the array of \[author_id, content_id\] sorted in the order of a_active where author_id is an author of content_id;
- a_emb - the array of author embeddings in 2018 sorted in the order of a_active;
- da_emb - the array of author embeddings in 2017 sorted in the order of a_active;
- a_edgellh - the list of the log likelihoods of forming author-author citation links sorted in the order of a_active.

```a_latest_cite_dists_2017.pkl```: <br />  
- a_latest_dists - the dictionary of (author_id, author_id's strategy distribution in 2017).

The three files in the ```pub_input``` folder are analogous.

### Training

You need to specify the hyperparameters in ```run.sh```.

If STRATEGY='cite', then NSTRATEGY=16; If STRATEGY='pub', then NSTRATEGY=8.

For the sample input, we set START_YEAR=2018 and END_YEAR=2019.

### Output

For the sample input: <br />  
- Training logs are stored in the ```run``` folder;
- The trained models are stored in ```2018_models.pkl``` in the ```cite_result``` folder;
- The strategy distribution of each content and the attention paid to each coauthor by a content are stored in ```2018_content_results.pkl``` in the ```cite_result``` folder;
- The strategy distribution of each author, the attention paid to each content by an author, and the attention paid by an author to his past are stored in ```2018_author_results.pkl``` in the ```cite_result``` folder.