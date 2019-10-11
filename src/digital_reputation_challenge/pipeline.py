# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pipeline construction."""

from kedro.pipeline import Pipeline, node
from digital_reputation_challenge.nodes.mynodes import (merge_df, app_split, cross_validation, cross_validation_train,
                                                        cross_validation_shap, find_hyperopt, cross_validation_test,
                                                        submit, lightfm_node, target_enc,kmean_node, naive_node,lda_node,
                                                        svd_node,umap_node)

# Here you can define your data-driven pipeline by importing your functions
# and adding them to the pipeline as follows:
#
# from nodes.data_wrangling import clean_data, compute_features
#
# pipeline = Pipeline([
#     node(clean_data, 'customers', 'prepared_customers'),
#     node(compute_features, 'prepared_customers', ['X_train', 'Y_train'])
# ])
#
# Once you have your pipeline defined, you can run it from the root of your
# project by calling:
#
# $ kedro run
#

def create_pipeline(**kwargs):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        Pipeline: The resulting pipeline.

    """

    lightfm = Pipeline([
        # node(lightfm_node, ["X1_train", "X2_train", "X1_test", "X2_test"], "lightfm_embed")
    ], name='lightfm')
    kmean_pipe = Pipeline([
        # node(kmean_node, ["X2_train","X2_test"], "kmean")
    ],name='kmean')

    naive_pipe = Pipeline([
        node(naive_node, ["X2_train","X2_test", "Y_train"], "naive")
    ],name='naive')

    lda_pipe = Pipeline([
        node(lda_node, ["X2_train","X2_test", "Y_train"], "lda")
    ],name='lda')

    svd_pipe = Pipeline([
        node(svd_node, ["X2_train","X2_test", "Y_train"], "svd")
    ],name='svd')

    umap_pipe = Pipeline([
        # node(umap_node, ["X2_train","X2_test", "Y_train"], "umap")
    ],name='umap')

    get_folds = Pipeline([
        node(app_split, "Y_train", "folds"),
        # node(merge_df,["X1_train","X2_train","X3_train","lightfm_embed","kmean","naive","lda","svd","Y_train"], "df_train_merge"),
        # node(merge_df,["X1_test","X2_test","X3_test","lightfm_embed","kmean","naive","lda","svd"], "df_test_merge"),
        # node(target_enc,["df_train_merge","df_test_merge"],["df_train","df_test"])
        node(merge_df,["X1_train","X2_train","X3_train","lightfm_embed","kmean","naive","lda","svd","umap","Y_train"], "df_train"),
        node(merge_df,["X1_test","X2_test","X3_test","lightfm_embed","kmean","naive","lda","svd","umap"], "df_test"),

    ], name='get_folds')

    hyper = Pipeline([
        # node(find_hyperopt, ["df_train","folds"], "hyper_best")
    ], name="hyper")

    pipeline = Pipeline([
        node(cross_validation, ["df_train", "folds", "hyper_best"], ["cv_result", "cv_models"]),
        node(cross_validation_train, ["df_train", "folds", "cv_models"], "cv_oof"),
        node(cross_validation_shap, ["df_train", "folds", "cv_models"], ["shap_reg", "shap_feature_stats"]),

    ], name="pipeline")

    test = Pipeline([
        node(cross_validation_test, ["df_test", "cv_models"], ["cv_test"]),
        node(submit, ["cv_test"], "submit")
    ], name="test")

    return umap_pipe+kmean_pipe +svd_pipe+lda_pipe+naive_pipe+ lightfm + get_folds + hyper+ pipeline + test
