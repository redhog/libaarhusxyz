import pandas as pd
import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
from pygltflib import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT, UNSIGNED_INT, VEC3, VEC4, SCALAR
from datetime import datetime
import struct
import io

from ..export.vtk import (
    _compute_xdist,
    _compute_sounding_widths,
    _generate_cells,
    _generate_points_array,
    _flatten_layer_data,
    _vtk_cell_data
)


def _create_gltf_buffer_data(point_coordinates, cell_indices_np, cells, attr_out):
    """Create binary buffer data for glTF file.

    Returns:
        buffer_data: bytes object containing all binary data
        accessors_info: list of (byte_offset, count, component_type, accessor_type) for each accessor
    """
    buffer_parts = []
    accessors_info = []
    current_offset = 0

    # 1. Position data (VEC3 FLOAT)
    positions = point_coordinates.astype(np.float32).flatten()
    positions_bytes = positions.tobytes()
    buffer_parts.append(positions_bytes)
    accessors_info.append({
        'offset': current_offset,
        'count': len(point_coordinates),
        'componentType': FLOAT,
        'type': VEC3,
        'min': point_coordinates.min(axis=0).tolist(),
        'max': point_coordinates.max(axis=0).tolist()
    })
    current_offset += len(positions_bytes)

    # Align to 4 bytes
    if current_offset % 4 != 0:
        padding = 4 - (current_offset % 4)
        buffer_parts.append(b'\x00' * padding)
        current_offset += padding

    # 2. Indices (SCALAR UNSIGNED_INT)
    # Convert quads to triangles (each quad = 2 triangles)
    triangles = []
    for quad in cell_indices_np:
        # Split quad [0,1,2,3] into triangles [0,1,2] and [0,2,3]
        triangles.append([quad[0], quad[1], quad[2]])
        triangles.append([quad[0], quad[2], quad[3]])

    indices = np.array(triangles, dtype=np.uint32).flatten()
    indices_bytes = indices.tobytes()
    buffer_parts.append(indices_bytes)
    accessors_info.append({
        'offset': current_offset,
        'count': len(indices),
        'componentType': UNSIGNED_INT,
        'type': SCALAR,
        'min': [int(indices.min())],
        'max': [int(indices.max())]
    })
    current_offset += len(indices_bytes)

    # Align to 4 bytes
    if current_offset % 4 != 0:
        padding = 4 - (current_offset % 4)
        buffer_parts.append(b'\x00' * padding)
        current_offset += padding

    # 3. Vertex colors from cell attributes (VEC4 FLOAT)
    # We'll create per-vertex colors by mapping cell data to vertices
    # Each vertex appears in multiple cells, so we'll average the values
    vertex_colors = np.zeros((len(point_coordinates), 4), dtype=np.float32)
    vertex_counts = np.zeros(len(point_coordinates), dtype=np.float32)

    # Use resistivity as the primary color attribute if available
    color_attr = 'resistivity' if 'resistivity' in cells.columns else (attr_out[0] if attr_out and attr_out[0] in cells.columns else None)

    if color_attr and color_attr in cells.columns:
        # Normalize attribute values to 0-1 range for color mapping
        values = cells[color_attr].values
        valid_mask = ~np.isnan(values)
        if valid_mask.any():
            vmin = values[valid_mask].min()
            vmax = values[valid_mask].max()
            if vmax > vmin:
                normalized = (values - vmin) / (vmax - vmin)
            else:
                normalized = np.zeros_like(values)
        else:
            normalized = np.zeros_like(values)

        # Map cell values to vertices
        for cell_idx, quad in enumerate(cell_indices_np):
            for vertex_idx in quad:
                # Simple colormap: blue (low) to red (high)
                val = normalized[cell_idx] if not np.isnan(normalized[cell_idx]) else 0.5
                vertex_colors[vertex_idx, 0] += val  # R
                vertex_colors[vertex_idx, 1] += 0.0   # G
                vertex_colors[vertex_idx, 2] += (1.0 - val)  # B
                vertex_colors[vertex_idx, 3] += 1.0   # A
                vertex_counts[vertex_idx] += 1

        # Average colors for shared vertices
        for i in range(len(point_coordinates)):
            if vertex_counts[i] > 0:
                vertex_colors[i, :3] /= vertex_counts[i]
                vertex_colors[i, 3] = 1.0  # Full opacity
    else:
        # Default gray color
        vertex_colors[:, :] = [0.5, 0.5, 0.5, 1.0]

    colors_bytes = vertex_colors.flatten().tobytes()
    buffer_parts.append(colors_bytes)
    accessors_info.append({
        'offset': current_offset,
        'count': len(point_coordinates),
        'componentType': FLOAT,
        'type': VEC4,
        'min': None,
        'max': None
    })
    current_offset += len(colors_bytes)

    return b''.join(buffer_parts), accessors_info


def _dump(model, fid, attr_out=['resistivity', 'resistivity_variance_factor', 'line_id', 'title', 'x', 'y',
                                  'topo', 'dep_top', 'dep_bot', 'tx_alt', 'invalt', 'invaltstd',
                                  'deltaalt', 'numdata', 'resdata', 'restotal', 'doi_upper', 'doi_lower', 'xdist']):
    """Export model to binary glTF (GLB) format.

    Similar to VTK export, creates a 3D mesh from resistivity model data.
    """
    # Handle infinite depth values (same as VTK)
    if np.isinf(model.layer_data["dep_bot"].iloc[:, -1].max()):
        print("WARNING: Last model.layer_data[dep_bot] contains inf values, "
              "falsifying the last dep_bot value. "
              "See .export.glb._dump for more details")
        last_layer_val = model.layer_data["dep_bot"].iloc[:, -2][0] + (
            model.layer_data["dep_bot"].iloc[:, -2][0] - model.layer_data["dep_bot"].iloc[:, -3][0])
        model.layer_data["dep_bot"] = model.layer_data["dep_bot"].replace([np.inf, -np.inf, np.nan], last_layer_val)

    fl = model.flightlines
    df = _flatten_layer_data(model)

    _compute_xdist(fl)
    _compute_sounding_widths(fl)
    cells = _generate_cells(fl, df)
    points_array = _generate_points_array(cells)

    point_coordinates, cell_indices_np, cells_out_vtk, cell_types_out_vtk = _vtk_cell_data(points_array)

    # Create glTF binary data
    buffer_data, accessors_info = _create_gltf_buffer_data(point_coordinates, cell_indices_np, cells, attr_out)

    # Build glTF structure
    gltf = GLTF2()

    # Buffer
    gltf.buffers = [Buffer(byteLength=len(buffer_data))]

    # Buffer views
    gltf.bufferViews = []
    position_accessor_info = accessors_info[0]
    indices_accessor_info = accessors_info[1]
    color_accessor_info = accessors_info[2]

    # Position buffer view
    position_size = point_coordinates.shape[0] * 3 * 4  # count * 3 floats * 4 bytes
    gltf.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=position_accessor_info['offset'],
        byteLength=position_size,
        target=ARRAY_BUFFER
    ))

    # Indices buffer view
    indices_size = indices_accessor_info['count'] * 4  # count * 4 bytes (UNSIGNED_INT)
    gltf.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=indices_accessor_info['offset'],
        byteLength=indices_size,
        target=ELEMENT_ARRAY_BUFFER
    ))

    # Color buffer view
    color_size = point_coordinates.shape[0] * 4 * 4  # count * 4 floats * 4 bytes
    gltf.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=color_accessor_info['offset'],
        byteLength=color_size,
        target=ARRAY_BUFFER
    ))

    # Accessors
    gltf.accessors = [
        # Position accessor
        Accessor(
            bufferView=0,
            byteOffset=0,
            componentType=position_accessor_info['componentType'],
            count=position_accessor_info['count'],
            type=position_accessor_info['type'],
            min=position_accessor_info['min'],
            max=position_accessor_info['max']
        ),
        # Indices accessor
        Accessor(
            bufferView=1,
            byteOffset=0,
            componentType=indices_accessor_info['componentType'],
            count=indices_accessor_info['count'],
            type=indices_accessor_info['type'],
            min=indices_accessor_info['min'],
            max=indices_accessor_info['max']
        ),
        # Color accessor
        Accessor(
            bufferView=2,
            byteOffset=0,
            componentType=color_accessor_info['componentType'],
            count=color_accessor_info['count'],
            type=color_accessor_info['type']
        )
    ]

    # Mesh
    gltf.meshes = [Mesh(
        primitives=[Primitive(
            attributes={'POSITION': 0, 'COLOR_0': 2},
            indices=1
        )]
    )]

    # Node
    gltf.nodes = [Node(mesh=0)]

    # Scene
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0

    # Set binary data
    gltf.set_binary_blob(buffer_data)

    # Write to file
    if isinstance(fid, io.IOBase):
        # For file-like objects, write the binary data
        glb_data = gltf.save_to_bytes()
        fid.write(glb_data)
    else:
        # fid is probably a file path string, but dump() handles that
        raise ValueError("_dump expects a file-like object, use dump() for file paths")


def dump(model, nameorfile, **kw):
    """Export model to binary glTF (GLB) format.

    Args:
        model: XYZ model instance with layer_data
        nameorfile: file path (string) or file-like object
        **kw: additional keyword arguments passed to _dump
    """
    if isinstance(nameorfile, str):
        # For file paths, use pygltflib's save mechanism
        # We need to create the GLTF2 object and save it
        # Create a temporary file-like object to capture the output
        buffer = io.BytesIO()
        _dump(model, buffer, **kw)
        buffer.seek(0)
        with open(nameorfile, 'wb') as f:
            f.write(buffer.read())
    else:
        return _dump(model, nameorfile, **kw)
