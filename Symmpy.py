import numpy as np

ERROR_CODE = {
    1: '[SymmPy]: One or more of the arguments do not have the appropriate shape.',
    2: '[SymmPy]: The magnitude of the rotation axis vector is zero.',
    3: '[SymmPy]: Provided plane values are collinear.',
    4: '[SymmPy]: One or more of the arguments are not the right data type.'
}


def __float_test(*values):
    for i in range(len(values)):
        try:
            if type(values[i]) == list:
                for j in range(len(values[i])):
                    float(values[i][j])
            else:
                float(values[i])
        except (ValueError, TypeError):
            return 0
        else:
            return 1


def clean(fractional_xyz, precision):
    """
    Calculates and returns the modulo 1 of a list of fractional coordinates.\n
    :rtype: list[float]
    :param list[float] fractional_xyz: The starting fractional coordinates\n
    :param int precision: The precision of the output coordinates\n
    :return: The modulo 1 of each member of the list\n
    """
    if len(fractional_xyz) != 3:
        raise IndexError(ERROR_CODE[1])
    test = __float_test(fractional_xyz, precision)
    if test == 0:
        raise ValueError(ERROR_CODE[4])
    formatted_fractional_xyz = [eval("{:.{dec}f}".format(i, dec=precision)) % 1 for i in fractional_xyz]
    new_fractional_xyz = [i % 1 for i in formatted_fractional_xyz]
    formatted_fractional_xyz = [eval("{:.{dec}f}".format(i, dec=precision)) % 1 for i in new_fractional_xyz]
    return formatted_fractional_xyz


def translate(xyz, directional_xyz):
    """
    Translates a vector defined in Cartesian space to another position\n
    :rtype: list
    :param list xyz: The starting vector\n
    :param list directional_xyz: The displacement vector\n
    :return: The translated vector\n
    """
    if len(xyz) != 3 or len(directional_xyz) != 3:
        raise IndexError(ERROR_CODE[1])
    test = __float_test(xyz, directional_xyz)
    if test == 0:
        raise ValueError(ERROR_CODE[4])
    start_vector = np.array([xyz[0], xyz[1], xyz[2], 1])
    translation_matrix = np.array([[1, 0, 0, directional_xyz[0]],
                                   [0, 1, 0, directional_xyz[1]],
                                   [0, 0, 1, directional_xyz[2]],
                                   [0, 0, 0, 1]])
    end_vector = list(np.delete(np.dot(translation_matrix, start_vector), -1))
    return end_vector


def rotate(xyz, rotation_axis_direction, angle, rotation_origin=[0, 0, 0]):
    """
    Rotate a vector in Cartesian space around an arbitrary axis\n
    :rtype: list
    :param list xyz:  The starting vector\n
    :param list rotation_axis_direction: Unit vector parallel to the rotation axis\n
    :param float angle: Angle of rotation in radians\n
    :param list rotation_origin: Any point through which the rotation axis passes
    :return: The rotated vector
    """
    if len(xyz) != 3 or len(rotation_axis_direction) != 3 or len(rotation_origin) != 3:
        raise IndexError(ERROR_CODE[1])
    test = __float_test(xyz, rotation_axis_direction, angle, rotation_origin)
    if test == 0:
        raise ValueError(ERROR_CODE[4])
    reverse_rotation_origin = list(-np.array(rotation_origin))
    start_vector = translate(xyz, reverse_rotation_origin)
    direction_vector = np.array([rotation_axis_direction[0], rotation_axis_direction[1], rotation_axis_direction[2]])
    if np.linalg.norm(direction_vector) != 0:
        normalised_direction_vector = direction_vector / np.linalg.norm(direction_vector)
        normalised_x = normalised_direction_vector[0]
        normalised_y = normalised_direction_vector[1]
        normalised_z = normalised_direction_vector[2]
        rotation_matrix = np.array([[np.cos(angle) + normalised_x ** 2 * (1 - np.cos(angle)),
                                     normalised_x * normalised_y * (1 - np.cos(angle)) - normalised_z * np.sin(angle),
                                     normalised_x * normalised_z * (1 - np.cos(angle)) + normalised_y * np.sin(angle)],
                                    [normalised_y * normalised_x * (1 - np.cos(angle)) + normalised_z * np.sin(angle),
                                     np.cos(angle) + normalised_y ** 2 * (1 - np.cos(angle)),
                                     normalised_y * normalised_z * (1 - np.cos(angle)) - normalised_x * np.sin(angle)],
                                    [normalised_z * normalised_x * (1 - np.cos(angle)) - normalised_y * np.sin(angle),
                                     normalised_z * normalised_y * (1 - np.cos(angle)) + normalised_x * np.sin(angle),
                                     np.cos(angle) + (normalised_z ** 2) * (1 - np.cos(angle))]])
        rotation_matrix = np.squeeze(rotation_matrix)
        rotated_vector = list(np.dot(rotation_matrix, start_vector))
        end_vector = translate(rotated_vector, rotation_origin)
        return end_vector
    else:
        raise ZeroDivisionError(ERROR_CODE[2])


def invert(xyz, inversion_origin=[0, 0, 0]):
    """
    Invert a vector's coordinates through an inversion centre located an arbitrary point \n
    :rtype: list
    :param list xyz: The starting vector\n
    :param list inversion_origin: The coordinates of the inversion centre\n
    :return: The inverted coordinates
    """
    if len(xyz) != 3 or len(inversion_origin) != 3:
        raise IndexError(ERROR_CODE[1])
    test = __float_test(xyz, inversion_origin)
    if test == 0:
        raise ValueError(ERROR_CODE[4])
    displacement_vector = list(-np.array(inversion_origin))
    start_vector = translate(xyz, displacement_vector)
    inversion_matrix = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]])
    inverted_vector = list(np.dot(inversion_matrix, start_vector))
    end_vector = translate(inverted_vector, inversion_origin)
    return end_vector


def rotoinvert(xyz, rotation_axis_direction, angle, rotoinversion_origin=[0, 0, 0]):
    """
    Rotate a vector around an arbitrary axis then invert its coordinates through an inversion centre that the rotation\n
    axis passes through
    :rtype: list
    :param list xyz: The starting vector\n
    :param list rotation_axis_direction: Unit vector parallel to the rotation axis\n
    :param float angle: Angle of rotation in radians\n
    :param list rotoinversion_origin:  The coordinates of the inversion centre\n
    :return: The vector after being transformed by an improper rotation
    """
    test = __float_test(xyz, rotation_axis_direction, angle, rotoinversion_origin)
    if test == 0:
        raise ValueError(ERROR_CODE[4])
    rotated_vector = rotate(xyz, rotation_axis_direction, angle, rotoinversion_origin)
    end_vector = invert(rotated_vector, rotoinversion_origin)
    return end_vector


def reflect(xyz, reflection_plane_coordinates):
    """
    Reflect a vector through an arbitrary plane
    :rtype: list
    :param list xyz: The starting vector\n
    :param list[list] reflection_plane_coordinates: A list with shape (3, 3) that contains three non-collinear points \n
    on the reflection plane\n
    :return: The reflected vector
    """
    test = __float_test(xyz,
                        reflection_plane_coordinates[0],
                        reflection_plane_coordinates[1],
                        reflection_plane_coordinates[2])
    if test == 0:
        raise ValueError(ERROR_CODE[4])
    displacement_vector = list(-np.array(reflection_plane_coordinates[0]))
    translated_vector = translate(xyz, displacement_vector)
    translated_coordinates = [translate(i, displacement_vector) for i in reflection_plane_coordinates]
    reflection_plane_vectors = np.array([[np.array(translated_coordinates[1]) -
                                          np.array(translated_coordinates[0])],
                                         [np.array(translated_coordinates[2]) -
                                          np.array(translated_coordinates[0])],
                                         [np.array(translated_coordinates[2]) -
                                          np.array(translated_coordinates[1])]])
    vector_norms = [np.linalg.norm(reflection_plane_vectors[0]),
                    np.linalg.norm(reflection_plane_vectors[1]),
                    np.linalg.norm(reflection_plane_vectors[2])]
    sorted_vector_norms = np.sort(vector_norms, axis=None)
    collinear = (sorted_vector_norms[0] + sorted_vector_norms[1] == sorted_vector_norms[2])
    if collinear:
            raise ValueError(ERROR_CODE[3])
    reflection_plane_normal_vector = np.cross(reflection_plane_vectors[0], reflection_plane_vectors[1])
    normalised_plane_normal_vector = reflection_plane_normal_vector / np.linalg.norm(
        reflection_plane_normal_vector)
    normalised_plane_normal_vector = np.squeeze(normalised_plane_normal_vector)
    z_rotation_axis_direction = np.cross(np.array([0, 0, 1]), normalised_plane_normal_vector)
    z_rotation_axis_direction = list(np.squeeze(z_rotation_axis_direction))
    z_rotation_angle = np.arccos(np.dot(normalised_plane_normal_vector, np.array([0, 0, 1])))
    rotated_vector = rotate(translated_vector, z_rotation_axis_direction, -z_rotation_angle)
    reflection_matrix = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, -1]])
    reflected_vector = list(np.dot(reflection_matrix, rotated_vector))
    rotated_reflected_vector = rotate(reflected_vector, z_rotation_axis_direction, z_rotation_angle)
    end_vector = translate(rotated_reflected_vector, reflection_plane_coordinates[0])
    return end_vector


def glide(xyz, directional_xyz, glide_plane_coordinates):
    """
    Reflect a vector through an arbitrary plane then translate in a desired direction parallel to the plane\n
    :rtype: list
    :param list xyz: The starting vector\n
    :param list directional_xyz: The direction in which to translate the vector\n
    :param list[list] glide_plane_coordinates: A list with shape (3, 3) that contains three non-collinear points \n
    on the glide plane\n
    :return:
    """
    reflected_vector = reflect(xyz, glide_plane_coordinates)
    end_vector = translate(reflected_vector, directional_xyz)
    return end_vector


def screw(xyz, directional_xyz, angle, screw_origin=[0, 0, 0]):
    """
    Rotate a vector around an arbitrary rotation axis, then translate in the direction parallel to the rotation axis\n
    :rtype: list
    :param list xyz: The starting vector\n
    :param list directional_xyz:  The direction of the rotation axis, and the direction in which to translate the vector
    \n
    :param float angle: The angle of rotation in radians\n
    :param list screw_origin: A point through which the screw axis passes\n
    :return:
    """
    rotated_vector = rotate(xyz, directional_xyz, angle, screw_origin)
    end_vector = translate(rotated_vector, directional_xyz)
    return end_vector
