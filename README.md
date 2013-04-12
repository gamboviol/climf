climf
=====

CLiMF Collaborative Less-is-More Filtering, a variant of latent factor CF
which optimises a lower bound of the smoothed reciprocal rank of "relevant"
items in ranked recommendation lists.  The intention is to promote diversity
as well as accuracy in the recommendations.  The method assumes binary
relevance data, as for example in friendship or follow relationships.

CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering
Yue Shi, Martha Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic
ACM RecSys 2012

To run on the supplied Epinions dataset:

    tar xzvf epinions.tar.gz
    python climf.py --train EP25_UPL5_train.mtx --test EP25_UPL5_test.mtx 
