import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
import copy


class SimpleCFNode:
    def __init__(self, max_instances, is_leaf=True, node_index=0, exemplar_model=None):
        self.is_leaf = is_leaf
        self.children = []
        self.exemplar_set = {}
        self.exemplar_model = exemplar_model
        self.max_instances = max_instances
        # 1 x feature_dim
        self.centroid = None
        self.classes = []
        self.new_data_points = 0
        self.node_index = node_index
        self.new = True

    def index_exemplar_set(self, index):
        return {
            key: value[index]
            for key, value in self.exemplar_set.items()
            if key != "finetuned_image_features"
        }

    def update_exemplar_set(self, data_points):
        # data_points is a dict
        self.new_data_points += len(data_points["image_features"])

        if not self.exemplar_set:
            self.exemplar_set = {key: [value] for key, value in data_points.items()}
        else:
            for key, value in data_points.items():
                assert key in self.exemplar_set, "Key not in exemplar set"
                self.exemplar_set[key].append(value)

        self.update_centroid(data_points["image_features"].unsqueeze(0))

    def update_centroid(self, data_point):
        data_point = torch.tensor(data_point)
        if self.centroid is None:
            self.centroid = data_point.mean(dim=0).unsqueeze(0)
        else:
            n = len(self.exemplar_set["image_features"])
            current_n = data_point.shape[0]
            self.centroid = (self.centroid * n + data_point.sum(dim=0).unsqueeze(0)) / (
                n + current_n
            )


class SimpleTree:
    def __init__(self, max_instances=50):
        self.root = SimpleCFNode(max_instances, is_leaf=True)
        self.max_instances = max_instances
        self.overall_node_index = 0

    def _find_closest_node(self, node, data_point):
        # data_point should be a dict at least containing "image_features"
        if node.is_leaf:
            return node
        min_distance = float("inf")
        closest_child = None
        for child in node.children:
            distance = torch.norm(
                child.centroid - torch.tensor(data_point["image_features"].unsqueeze(0))
            )
            if distance < min_distance:
                min_distance = distance
                closest_child = child
        return self._find_closest_node(closest_child, data_point)

    def insert(self, data_point):
        closest_node = self._find_closest_node(self.root, data_point)
        closest_node.update_exemplar_set(data_point)

        if len(closest_node.exemplar_set["image_features"]) > self.max_instances:
            self._split_node(closest_node)

    def fit_data_into_tree(self, data_points):
        # data_points should be a dict, we will patch each data point into the tree
        # get the length of the data points
        data_points_len = len(data_points["image_features"])
        for i in tqdm(range(data_points_len)):
            data_point = {key: value[i] for key, value in data_points.items()}
            # new has set to be false after trained during the sleep phase
            data_point["new"] = True
            data_point["sample_weights"] = 0.99
            data_point["sample_counts"] = 0
            # print("Inserting data point", data_point)
            self.insert(data_point)

    def _split_node(self, node):
        kmeans = KMeans(n_clusters=2, random_state=0)
        # TODO: we should use the raw image features here
        data_points_np = torch.stack(
            node.exemplar_set["raw_image_features"], dim=0
        ).numpy()

        kmeans.fit(data_points_np)

        self.overall_node_index += 1
        child1 = SimpleCFNode(
            self.max_instances,
            node_index=self.overall_node_index,
            exemplar_model=copy.deepcopy(node.exemplar_model),
        )
        self.overall_node_index += 1
        child2 = SimpleCFNode(
            self.max_instances,
            node_index=self.overall_node_index,
            exemplar_model=copy.deepcopy(node.exemplar_model),
        )

        for i, label in enumerate(kmeans.labels_):
            if label == 0:
                # pack the data point into child1
                child1.update_exemplar_set(node.index_exemplar_set(i))
            else:
                child2.update_exemplar_set(node.index_exemplar_set(i))

        node.is_leaf = False
        node.children = [child1, child2]
        node.exemplar_model = None
        node.exemplar_set = {}
        node.new_data_points = 0

        # release the memory of the exemplar model
        torch.cuda.empty_cache()

    def get_trainable_nodes(self):
        leaf_nodes = self.get_leaf_nodes(self.root)
        return [node for node in leaf_nodes if node.new_data_points > 0]

    def get_leaf_nodes(self, node):
        if node.is_leaf:
            return [node]
        leaf_nodes = []
        for child in node.children:
            leaf_nodes.extend(self.get_leaf_nodes(child))
        return leaf_nodes

    def get_all_leaf_nodes(self):
        return self.get_leaf_nodes(self.root)

    def get_models_and_centroids(self):
        leaf_nodes = self.get_leaf_nodes(self.root)
        return (
            [leaf.exemplar_model for leaf in leaf_nodes],
            [leaf.centroid for leaf in leaf_nodes],
        )
