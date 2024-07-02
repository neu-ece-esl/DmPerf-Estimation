# Extract raw data from a PDF file (which needs to be in vector format)
# Implemented with lots of help from ChatGPT and Copilot (in VSCode)
# Procedure:
#   1 - Open the PDF file in Inkscape (potentially ungroup objects)
#   2 - Save the file as an SVG file
#   3 - Identify the lines (or tick marks) to get known y values from location in graph
#   4 - Identify the paths of measurement lines to extract the values from
#   5 - Update the script with the path ids and line ids (pathList and lineList dictionaries), update the y values
#   6 - Run the script ... and debug from there


import xml.etree.ElementTree as ET
import numpy as np
from svgpathtools import parse_path

def parse_transform(transform_str):
    """
    Parses a transform attribute string and returns a transformation matrix.
    Only handles translate and matrix transformations for simplicity.
    """
    transform = np.eye(3)  # Identity matrix (no transformation)
    parts = transform_str.split(')')
    for part in parts:
        if 'translate' in part:
            values = part.split('(')[1].split(',')
            tx = float(values[0])
            ty = float(values[1])
            translation_matrix = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
            transform = np.dot(translation_matrix, transform)
        elif 'matrix' in part:
            values = part.split('(')[1].split(',')
            matrix = np.array([
                [float(values[0]), float(values[2]), float(values[4])],
                [float(values[1]), float(values[3]), float(values[5])],
                [0, 0, 1]
            ])
            transform = np.dot(matrix, transform)
    
    return transform

def apply_transform(point, transform_matrix):
    """
    Applies the transformation matrix to a point and returns the transformed point.
    """
    point_homogeneous = np.array([point.real, point.imag, 1])
    transformed_point = np.dot(transform_matrix, point_homogeneous)
    return (transformed_point[0], transformed_point[1])

def extract_paths(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    paths_info = []
    
    # Look for path elements
    paths = root.findall('.//{http://www.w3.org/2000/svg}path')
    for path in paths:
        path_id = path.attrib.get('id', None)
        d = path.attrib['d'].strip()
        transform_str = path.attrib.get('transform', None)
        
        # Parse the path using svgpathtools to get the points
        path_obj = parse_path(d)
        if len(path_obj) == 0:
            continue  # Skip empty paths
        
        points = []
        for segment in path_obj:
            for i in range(len(segment)):
                points.append(segment.point(i))
        
        transform_matrix = np.eye(3)
        
        if transform_str:
            transform_matrix = parse_transform(transform_str)
        
        # Apply transform to each point
        transformed_points = [apply_transform(point, transform_matrix) for point in points]
        
        paths_info.append({
            'id': path_id,
            'points': transformed_points
        })
    
    return paths_info

# Example usage
svg_file = 'OpenVXperf.svg'
paths_info = extract_paths(svg_file)

pathList = { 
    'path98': 'OOP',
    'path109': 'RDP',
    'path120': 'GIDE',
    'path131': 'GIDE1:1',
    'path142': 'FOP',
}

lineList = {
    'path96': '16.5-19',
    'path97': '16'
}


# Print for lineList (complete lines)
for line_info in paths_info:
    if line_info['id'] in lineList:
        print(f"{lineList[line_info['id']]}")
        if lineList[line_info['id']] == '16':
            # store y of first point
            y16 = line_info['points'][0][1]
        if lineList[line_info['id']] == '16.5-19':
            # store y of last point
            y19 = line_info['points'][-1][1]
            
        for point in line_info['points']:
            print(f"{point[0]}, {point[1]}")

# Print path id and all points for each path
for path_info in paths_info:
    if path_info['id'] in pathList:
        print(f"{pathList[path_info['id']]}")
        pointNr = 0
        for point in path_info['points']:
            # since lines are continuous skip the even points unless it is the last
            if pointNr % 2 == 0 or pointNr == len(path_info['points']) - 1:
                # compute the y value of the point based on y16 and y19's difference
                yVal = (point[1] - y16) * ((19-16)/(y19 - y16)) + 16
                print(f"{yVal}, {point[1]}")

                #print(f"{point[0]}, {point[1]}")
            pointNr += 1
            

