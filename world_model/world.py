from frame import frame as frame_t
import world_model.object_projection as object_projection_
import numpy as np
from sklearn import mixture


class World:
    def __init__(self):
        self.objects_projection = []
        self.objects = []

    def add_objects_projection_from_frame(self, frame, confidence):
        self.objects_projection.extend(frame.get_objects_projection_with_confidence_more_than(confidence))

    def unify_objects_projection_get_object(self):
        print('doing unify')
        map_class_to_projection = {}
        for projection in self.objects_projection:
            # print(projection)
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
            # print(len(projection_position), 'size', len(projection_bayesian_gaussian_mixture.means_))

            projections_index = projection_bayesian_gaussian_mixture.predict(projection_position)
            projection_mixture_means = projection_bayesian_gaussian_mixture.means_
            projection_mixture_precision = projection_bayesian_gaussian_mixture.precisions_
            if not projection_bayesian_gaussian_mixture.converged_:
                print('FATAL: object projections cannot converge, World.objects is not generated')
                return

            print(projections_index)

            cluster_index_set = set(projections_index)
            print(cluster_index_set)
            projection_cluster = {}
            for i in cluster_index_set:
                projection_cluster[i] = []

            for i in range(len(projections)):
                # print('projections len', len(projections))
                # print('projections_index len', len(projections_index))
                # print('projection_cluster len', len(projection_cluster))
                projection_cluster[projections_index[i]].append(projections[i])

            for cluster_index in projection_cluster.keys():
                projections_group = projection_cluster[cluster_index]
                print('-----------start------------')
                for projection in projections_group:
                    print(projection)
                print('-----------end------------')
                frame_list = list(map(lambda e: e['frame_id'], projections_group))
                frame_set = set(frame_list)
                class_list = list(map(lambda e: e['class'], projections_group))
                class_set = set(class_list)

                if len(class_set) != 1:
                    print('FATAL: in a projection cluster, every projection should have the same class.')

                if len(frame_list) != len(frame_set):
                    print('WARNING! a projection cluster contains multiple projections in one frame')
                    print(class_set)
                    max_repeat = max([frame_list.count(i) for i in frame_set])
                    print(max_repeat)
                    projection_group_gaussian_mixture = mixture.GaussianMixture(n_components=max_repeat,
                                                                                covariance_type='full')
                    projections_group_position = list(map(lambda e: e['position'], projections_group))
                    projection_group_gaussian_mixture.fit(projections_group_position)

                    if not projection_group_gaussian_mixture.converged_:
                        # TODO: handle this later
                        print('FATAL: object projections cannot converge, World.objects is not generated')

                    projection_group_mixture_means = projection_group_gaussian_mixture.means_
                    projection_group_mixture_precisions = projection_group_gaussian_mixture.precisions_

                    projection_group_index_list = projection_group_gaussian_mixture.predict(projections_group_position)
                    projection_group_index_set = set(projection_group_index_list)

                    if len(projection_group_index_set) < max_repeat:
                        print('WARNING: fail to distinguish the object has the same class')
                    for i in range(max_repeat):
                        new_objects.append(
                            World.new_object(projection_group_mixture_means[i],
                                             projection_group_mixture_precisions[i],
                                             list(class_set)[0]))
                else:
                    new_objects.append(
                        World.new_object(projection_mixture_means[cluster_index],
                                         projection_mixture_precision[cluster_index],
                                         list(class_set)[0]))

        self.objects = new_objects

    @staticmethod
    def new_object(position, precision, clazz):
        return {'position': position, 'precision': precision, 'class': clazz}
