Elman Net: ``d_emb`` vs ``d_hid`` vs ``n_lyr``
==============================================

Abstract
--------
This goal of this experiment is to show how Elman Net model structure hyperparameters affect training loss and perplexity.
We found that

- Increasing ``d_emb`` makes training loss and perplexity lower.
- Increasing ``d_hid`` in general makes training loss and perplexity lower.
- No general conclusion can be made when increasing ``n_lyr``.
- Lower training loss does not guarentee lower perplexity.
- All experiments are underfitting.

Environment setup
-----------------
We run experiments on Nvidia RTX 2070S.
CUDA version is ``11.4`` and CUDA driver version is ``470.129.06``.

Experiment setup
----------------
We change the values of ``d_emb``, ``d_hid`` and ``n_lyr`` and record training loss and perplexity.
Parameters and their values are list below.

+-----------+-----------------------+
| Name      | Values                |
+===========+=======================+
| ``d_emb`` | :math:`\set{10, 100}` |
+-----------+-----------------------+
| ``d_hid`` | :math:`\set{10, 100}` |
+-----------+-----------------------+
| ``n_lyr`` | :math:`\set{1, 2, 3}` |
+-----------+-----------------------+

Tokenizer settings
~~~~~~~~~~~~~~~~~~
We use character tokenizer :py:class:`lmp.tknzr.CharTknzr`.
We use :py:mod:`lmp.script.train_tknzr` to train our tokenizer.
Script was called as below:

.. code-block:: shell

  python -m lmp.script.train_tknzr character \
    --dset_name demo \
    --exp_name demo_tknzr \
    --is_uncased \
    --max_vocab -1 \
    --min_count 0 \
    --ver train

Model training settings
~~~~~~~~~~~~~~~~~~~~~~~
We train Elman Net language model :py:class:`lmp.model.ElmanNet` with different model structure hyperparameters.
We use :py:mod:`lmp.script.train_model` to train language models.
Script was called as below:

.. code-block:: shell

  python -m lmp.script.train_model Elman-Net \
    --dset_name demo \
    --batch_size 32 \
    --beta1 0.9 \
    --beta1 0.99 \
    --ckpt_step 500 \
    --ctx_win 16 \
    --d_emb D_EMB \
    --d_hid D_HID \
    --dset_name demo \
    --eps 1e-8 \
    --exp_name EXP_NAME \
    --log_step 100 \
    --lr 1e-3 \
    --max_norm 1 \
    --max_seq_len 35 \
    --n_lyr N_LYR \
    --p_emb 0.0 \
    --p_hid 0.0 \
    --tknzr_exp_name demo_tknzr \
    --total_step 30000 \
    --ver train \
    --warmup_step 5000 \
    --wd 1e-2

Model evaluation settings
~~~~~~~~~~~~~~~~~~~~~~~~~
We evaluate language model using :py:mod:`lmp.script.eval_dset_ppl`.
Script was called as below:

.. code-block:: shell

  python -m lmp.script.eval_dset_ppl demo \
    --batch_size 512 \
    --first_ckpt 0 \
    --exp_name EXP_NAME \
    --ver VER

Experiment results
------------------
All results were logged on tensorboard.
You can launch tensorboard with the script

.. code-block:: shell

  pipenv run tensorboard

Training loss
~~~~~~~~~~~~~

+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| ``d_emb`` | ``d_hid`` | ``n_lyr`` | 5k steps   | 10k steps  | 15k steps  | 20k steps  | 25k steps  | 30k steps  |
+===========+===========+===========+============+============+============+============+============+============+
| 10        | 10        | 1         | 0.6797     | 0.3004     | 0.2687     | 0.2676     | 0.2642     | 0.2603     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 10        | 2         | 0.5803     | 0.2753     | 0.2617     | 0.2634     | 0.2586     | 0.2551     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 10        | 3         | 0.9195     | 0.3842     | 0.3111     | 0.2963     | 0.2861     | 0.2824     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 100       | 1         | 0.7075     | 0.2624     | 0.254      | 0.2579     | 0.2542     | 0.2515     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 100       | 2         | 1.419      | 0.2982     | 0.2575     | 0.2597     | 0.2565     | 0.2530     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 10        | 100       | 3         | 1.08       | 0.2599     | 0.254      | 0.2577     | 0.2542     | 0.2519     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 10        | 1         | 0.2655     | 0.2572     | 0.2545     | 0.2591     | 0.2553     | 0.2497     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 10        | 2         | 0.2666     | 0.2556     | 0.2527     | 0.2567     | 0.254      | 0.2509     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 10        | 3         | 0.2646     | 0.2526     | 0.2476     | 0.2536     | 0.2452     | 0.2409     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 100       | 1         | 0.2448     | 0.2181     | 0.2056     | 0.2016     | 0.1941     | 0.1856     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 100       | 2         | **0.2272** | **0.2153** | 0.205      | **0.1984** | **0.1883** | 0.1759     |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+
| 100       | 100       | 3         | 0.2439     | 0.2197     | **0.2049** | 0.2        | 0.1934     | **0.1784** |
+-----------+-----------+-----------+------------+------------+------------+------------+------------+------------+

Observation 1: Increasing ``d_emb`` from ``10`` to ``100`` makes training loss smaller.
***************************************************************************************
By fixing ``d_hid`` and ``n_lyr``, we compare training loss for ``d_emb = 10`` and ``d_emb = 100``.
All comparisons (:math:`\dfrac{36}{36}`) show that training loss is smaller when increasing ``d_emb``.
Thus we conclude that increasing ``d_emb`` from ``10`` to ``100`` makes training loss smaller.

Observation 2: Increasing ``d_hid`` from ``10`` to ``100`` in general makes training loss smaller.
**************************************************************************************************
By fixing ``d_emb`` and ``n_lyr``, we compare training loss for ``d_hid = 10`` and ``d_hid = 100``.
:math:`32` out of :math:`36` comparisons show that training loss is smaller when increasing ``d_hid``.
This suggest that increasing ``d_hid`` from ``10`` to ``100`` in general makes training loss smaller.

Observation 3: Increasing ``n_lyr`` has not significant behavior.
*****************************************************************
By fixing ``d_emb`` and ``d_hid``, we compare training loss for ``n_lyr = 1`` and ``n_lyr = 2``.
Only :math:`16` out of :math:`24` comparisons show that training loss is smaller when increasing ``n_lyr``.
Increasing ``n_lyr`` further (from ``1`` to ``3``) does not make training loss smaller, neither.
Only :math:`13` out of :math:`24` comparisons show that training loss is smaller when increasing ``n_lyr``.
No significant results can be concluded.

Observation 4: When ``d_emb = 10`` and ``d_hid = 10``, increasing ``n_lyr`` shows inconsistent behavior.
********************************************************************************************************
This is a further observation of Observation 3.
By fixing ``d_emb = 10`` and ``d_hid = 10``, we compare training loss for ``n_lyr = 1`` and ``n_lyr = 2``.
Increasing ``n_lyr`` from ``1`` to ``2`` makes training loss smaller (:math:`\dfrac{6}{6}`).
But increasing ``n_lyr`` further to ``3`` makes training loss larger (:math:`\dfrac{6}{6}`).

Observation 5: When ``d_emb = 10`` and ``d_hid = 100``, increasing ``n_lyr`` in general makes training loss larger.
*******************************************************************************************************************
This is a further observation of Observation 3.
By fixing ``d_emb = 10`` and ``d_hid = 100``, we compare training loss for ``n_lyr = 1`` and ``n_lyr = 2``.
All comparisons (:math:`\dfrac{6}{6}`) show that training loss is larger when increasing ``d_emb``.
But increasing ``n_lyr`` further (from ``1`` to ``3``) has a three-way tie (:math:`\dfrac{2}{6}` in all cases).
No conclusion can be made for the last case.

Observation 6: When ``d_emb = 100``, increasing ``n_lyr`` in general makes training loss smaller.
*************************************************************************************************
This is a further observation of Observation 3.
By fixing ``d_emb = 100`` and ``d_hid``, we compare training loss for ``n_lyr = 1`` and ``n_lyr = 2``.
:math:`10` out of :math:`12` comparisons show that training loss is smaller when increasing ``n_lyr``.
Increasing ``n_lyr`` further (from ``1`` to ``3``) has similar behavior.
:math:`11` out of :math:`12` comparisons show that training loss is smaller when increasing ``n_lyr``.
Thus we conclude that when ``d_emb = 100``, increasing ``n_lyr`` from ``1`` to ``2`` or ``3`` in general makes training loss smaller.

Observation 7: Increasing ``n_lyr`` must also increase ``d_emb``.
*****************************************************************
Combining observations in 3 and 6, it suggest that when increasing ``n_lyr`` one have to increase ``d_emb`` together to make training loss smaller.

Perplexity
~~~~~~~~~~

+-----------+-----------+-----------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
| ``d_emb`` | ``d_hid`` | ``n_lyr`` | 5k steps                          | 10k steps                         | 15k steps                         | 20k steps                         | 25k steps                         | 30k steps                         |
|           |           |           +-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|           |           |           | train     | valid     | test      | train     | valid     | test      | train     | valid     | test      | train     | valid     | test      | train     | valid     | test      | train     | valid     | test      |
+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+
| 10        | 10        | 1         | 4.018     | 3.843     | 4.088     | 4.3       | 4.477     | 5.604     | 5.089     | 6.87      | 9.125     | 5.509     | 10.39     | 11.49     | 6.269     | 11.92     | 15.63     | 6.193     | 11.45     | 15.51     |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 10        | 2         | 3.982     | 3.93      | 4.159     | 7.744     | 8.173     | 9.336     | 9.768     | 10.73     | 12.38     | 11.51     | 12.93     | 15.01     | 13.95     | 17.76     | 19.22     | 13.97     | 20.53     | 21.09     |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 10        | 3         | 6.264     | 5.988     | 6.717     | 4.343     | 4.243     | 4.997     | 6.022     | 5.515     | 7.005     | 6.083     | 5.624     | 7.152     | 5.967     | 5.551     | 7.047     | 5.948     | 5.538     | 7.052     |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 100       | 1         | 15.31     | 13.51     | 14.48     | 17.59     | 26.52     | 25.85     | 22.3      | 41.9      | 38.11     | 29.22     | 54.44     | 49.2      | 34.27     | 58.01     | 55.97     | 35.77     | 61.97     | 59.23     |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 100       | 2         | 4.966     | 4.97      | 5.185     | 3.73      | 5.63      | 5.664     | 4.588     | 7.856     | 8.175     | 5.477     | 9.059     | 9.221     | 5.84      | 10.03     | 10.7      | 5.508     | 9.083     | 10.84     |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 10        | 100       | 3         | 6.595     | 6.539     | 7.029     | 4.432     | 5.727     | 6.07      | 5.102     | 11.02     | 9.23      | 5.019     | 12.7      | 9.523     | 7.362     | 22.49     | 15.26     | 7.648     | 24.01     | 15.71     |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 10        | 1         | **1.908** | 3.488     | 3.363     | **1.952** | 4.906     | 4.121     | **1.956** | 5.859     | 4.569     | **1.999** | 6.75      | 4.96      | **2.188** | 7.108     | 5.354     | **2.356** | 6.069     | 5.02      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 10        | 2         | 6.111     | 7.492     | 8.515     | 14.74     | 20.2      | 22.35     | 17.99     | 25.92     | 28.24     | 17.49     | 25.42     | 27.74     | 17.68     | 25.85     | 28.29     | 17.74     | 26.2      | 28.54     |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 10        | 3         | 14.63     | 12.38     | 19.23     | 22.73     | 22.84     | 31.88     | 16.78     | 17.8      | 22.16     | 20.7      | 29.82     | 32.95     | 23.67     | 36.28     | 39.5      | 26.19     | 46.51     | 47.41     |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 100       | 1         | 1.973     | **2.852** | **2.71**  | 1.964     | **2.803** | **2.952** | 2.278     | **3.138** | **3.347** | 2.4       | **3.486** | **3.493** | 2.51      | **3.78**  | **3.606** | 2.761     | **4.136** | **3.958** |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 100       | 2         | 2.352     | 2.996     | 3.287     | 3.153     | 3.791     | 4.143     | 4.231     | 5.226     | 5.794     | 5.232     | 6.432     | 7.074     | 4.833     | 6.33      | 6.718     | 4.973     | 6.591     | 6.86      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 100       | 100       | 3         | 3.86      | 4.436     | 5.248     | 3.268     | 4.05      | 4.488     | 3.119     | 4.434     | 4.581     | 4.087     | 5.606     | 5.724     | 4.285     | 5.923     | 6.036     | 4.578     | 6.311     | 6.376     |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+

Observation 1: Increasing ``d_emb`` in general makes perplexity smaller.
************************************************************************
By fixing ``d_hid`` and ``n_lyr``, we compare perplexity for ``d_emb = 10`` and ``d_emb = 100``.
Most comparisons (:math:`\dfrac{71}{108}`) show that perplexity is smaller when increasing ``d_emb``.

Observation 2: When ``d_hid = 10`` and ``n_lyr = 2 or 3``, increasing ``d_emb`` makes perplexity larger.
********************************************************************************************************
This is a further observation of Observation 1.
By fixing ``d_hid = 10`` and ``n_lyr = 2 or 3``, we compare perplexity for ``d_emb = 10`` and ``d_emb = 100``.
All comparisons (:math:`\dfrac{36}{36}`) show that perplexity is larger when increase ``d_emb``.

Observation 3: Increasing ``d_hid`` in general makes perplexity smaller.
************************************************************************
By fixing ``d_emb`` and ``n_lyr``, we compare perplexity for ``d_hid = 10`` and ``d_hid = 100``.
More than half of the comparisons (:math:`\dfrac{63}{108}`) show that perplexity is smaller when increasing ``d_hid``.

Observation 4: When ``d_emb = 10`` and ``n_lyr = 1 or 3``, increasing ``d_hid`` makes perplexity larger.
********************************************************************************************************
This is a further observation of Observation 3.
By fixing ``d_emb = 10`` and ``n_lyr = 1 or 3``, we compare perplexity for ``d_hid = 10`` and ``d_hid = 100``.
Almost all comparisons (:math:`\dfrac{18}{18}` and :math:`\dfrac{17}{18}`) show that perplexity is larger when increasing ``d_hid``.

Observation 5: Increasing ``n_lyr`` in general makes perplexity larger.
***********************************************************************
By fixing ``d_emb`` and ``n_lyr``, we compare perplexity for ``n_lyr = 1`` and ``n_lyr = 2``.
Most comparisons (:math:`\dfrac{54}{72}`) show that perplexity is larger when increasing ``n_lyr``.
Increasing ``n_lyr`` further (from ``1`` to ``3``) has similar behavior.
More than half of comparisons (:math:`\dfrac{42}{72}`) show that perplexity is larger when increasing ``n_lyr``.

Observation 6: Increasing ``n_lyr`` from ``1`` to ``3`` has inconsistent behavior.
**********************************************************************************
This is a further observation of Observation 5.
We fix ``d_hid`` and compare perplexity for ``n_lyr = 1`` and ``n_lyr = 3``.
When ``d_emb = 10``, most comparisons (:math:`\dfrac{30}{36}`) show that perplexity is smaller when increasing ``n_lyr``.
But when ``d_emb = 100``, all comparisons (:math:`\dfrac{36}{36}`) show that perplexity is larger when increasing ``n_lyr``.

Observation 7: Elman Net language models may still underfitting.
****************************************************************
For all configuration, perplexity has the increasing tendency across all dataset.
This is unexpected when loss is convergent.

Observation 8: Low perplexity happens at ``5k`` steps.
******************************************************
We use ``--warmup_step 5000`` to train our language model.
This might suggest that we use larger ``--warmup_step`` to tune Elman Net language models.
By observation 7, it seems that ``--total_step`` does not need to adjust.

Future work
-----------
We will try to make Elman Net overfitting.
We will do it by increasing ``--warmup_step`` and adding dropout.
