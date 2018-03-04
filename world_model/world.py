import numpy as np
from sklearn import mixture


def new_object(position, clazz, orientation, size):
    return {'position': position, 'class': clazz, 'orientation': orientation, 'size': size}


class World:
    def __init__(self):
        self.objects_projection = []
        self.objects = []

    def add_projections(self, projections):
        self.objects_projection.extend(projections)

    def unify_objects_projection_get_object(self):
        map_class_to_projection = {}
        for projection in self.objects_projection:
            projection_class = projection['class']
            if not (projection_class in map_class_to_projection.keys()):
                map_class_to_projection[projection_class] = []
            map_class_to_projection[projection_class].append(projection)

        new_objects = []

        for projections in map_class_to_projection.values():

            if len(projections) == 1:
                print('WARNING: single projection in a class')
                continue
                # TODO: handle this later

            projection_position = list(map(lambda e: e['position'], projections))

            projection_bayesian_gaussian_mixture = mixture.BayesianGaussianMixture(
                n_components=len(projection_position) // 2 + 1,
                covariance_type='full')
            projection_bayesian_gaussian_mixture.fit(projection_position)

            projections_index = projection_bayesian_gaussian_mixture.predict(projection_position)
            projection_mixture_means = projection_bayesian_gaussian_mixture.means_
            if not projection_bayesian_gaussian_mixture.converged_:
                print('FATAL: object projections cannot converge, World.objects is not generated')
                return

            projection_cluster = World.generate_projection_cluster(projections, projections_index)

            cluster_max_size = max(list(map(lambda e: len(e), projection_cluster.values())))

            abandoning_index = []
            for i, c in projection_cluster.items():
                size = len(c)
                if size / cluster_max_size < 0.5:
                    abandoning_index.append(i)

            for i in abandoning_index:
                projection_cluster.pop(i)

            for cluster_index in projection_cluster.keys():
                projections_group = projection_cluster[cluster_index]
                frame_list = list(map(lambda e: e['frame_id'], projections_group))
                frame_set = set(frame_list)
                class_list = list(map(lambda e: e['class'], projections_group))
                class_set = set(class_list)

                if len(class_set) != 1:
                    print('FATAL: in a projection cluster, every projection should have the same class.')

                if len(frame_list) != len(frame_set):
                    print('WARNING! a projection cluster contains multiple projections in one frame')
                    # print(class_set)
                    max_repeat = max([frame_list.count(i) for i in frame_set])
                    # print(max_repeat)
                    projection_group_gaussian_mixture = mixture.GaussianMixture(n_components=max_repeat,
                                                                                covariance_type='full')
                    projections_group_position = list(map(lambda e: e['position'], projections_group))
                    projection_group_gaussian_mixture.fit(projections_group_position)

                    predictions = projection_group_gaussian_mixture.predict(projections_group)

                    projections_group_cluster = World.generate_projection_cluster(projections_group, predictions)

                    if not projection_group_gaussian_mixture.converged_:
                        # TODO: handle this later
                        print('FATAL: object projections cannot converge, World.objects is not generated')

                    projection_group_mixture_means = projection_group_gaussian_mixture.means_

                    projection_group_index_list = projection_group_gaussian_mixture.predict(projections_group_position)
                    projection_group_index_set = set(projection_group_index_list)

                    if len(projection_group_index_set) < max_repeat:
                        print('WARNING: fail to distinguish the object has the same class')

                    for i, projs in projections_group_cluster.items():
                        orientation, size = World.generate_orientation_size_from_projections(projs)
                        new_objects.append(
                            new_object(list(projection_group_mixture_means[i]),
                                       list(class_set)[0], orientation, size))
                else:
                    orientation, size = World.generate_orientation_size_from_projections(projections_group)
                    new_objects.append(
                        new_object(list(projection_mixture_means[cluster_index]),
                                   list(class_set)[0], orientation, size))

        self.objects = new_objects

    @staticmethod
    def generate_orientation_size_from_projections(projections):
        orientation = sum(list(map(lambda e: e['orientation'] / np.linalg.norm(e['orientation']), projections))) / len(
            projections)
        orientation = orientation / np.linalg.norm(orientation)
        size = np.mean(np.array(list(map(lambda e: e['size'], projections))))
        return orientation, size

    @staticmethod
    def generate_projection_cluster(projections, predictions):
        prediction_index_set = set(predictions)

        projection_cluster = {}

        for prediction_index in prediction_index_set:
            projection_cluster[prediction_index] = []

        for i, projection in enumerate(projections):
            projection_cluster[predictions[i]].append(projection)

        return projection_cluster
