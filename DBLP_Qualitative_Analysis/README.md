## DBLP Qualitative Analysis

An empirical application of our framework to the DBLP academic dataset for qualitative analysis.

We use the <a href="https://aminer.org/citation">DBLP-Citation-network V11</a> dataset. The data processing codes are provided for readers' reference.

In particular, we aim to discover strategic behavior associated with two paper attributes: citations and publication venues. That is, *what are the strategic considerations behind whom to cite, and where to publish?* 

We set papers created during the years 1980-1999 as the background papers and examine the strategies adopted by authors who publish papers between 2000 and 2018 inclusive. We infer an author or paper's citation or location strategies only if we can observe the corresponding citation or location edges. This corresponds to 97% of the papers in the dataset.