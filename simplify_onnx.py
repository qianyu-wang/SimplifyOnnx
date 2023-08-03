from pathlib import Path

import numpy as np
import onnx
import onnxsim
import onnxruntime
import streamlit as st
from easydict import EasyDict


def simplify(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    简化ONNX模型，去除无用的节点和参数

    Args:
        model (onnx.ModelProto): 需要简化的ONNX模型

    Raises:
        Exception: 简化ONNX模型失败

    Returns:
        onnx.ModelProto: 简化后的ONNX模型
    """
    model_sim, success = onnxsim.simplify(model)
    if not success:
        raise Exception("简化ONNX模型失败！")
    return model_sim


def modify_reshape(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    重新编排ONNX模型中的Reshape节点，使得batch维度为-1，其它维度为固定值。

    Args:
        model (onnx.ModelProto): 需要重新编排Reshape节点的ONNX模型

    Returns:
        onnx.ModelProto: 重新编排后的ONNX模型
    """
    # 复制一份模型，防止修改原模型
    model = onnx.load_model_from_string(model.SerializeToString())

    # 运行模型，获取每个Reshape节点的输出维度
    temp_model = onnx.load_model_from_string(model.SerializeToString())
    reshape_names = []
    for node in temp_model.graph.node:
        if node.op_type == "Reshape":
            temp_model.graph.output.extend(
                [
                    onnx.helper.make_tensor_value_info(
                        node.output[0], onnx.TensorProto.FLOAT, [None]
                    )
                ]
            )
            reshape_names.append(node.output[0])
    input_name = temp_model.graph.input[0].name
    input_shape = [
        dim.dim_value for dim in temp_model.graph.input[0].type.tensor_type.shape.dim
    ]
    input_data = np.random.random_sample(input_shape).astype(np.float32)
    sess = onnxruntime.InferenceSession(
        temp_model.SerializeToString(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    outputs = sess.run(reshape_names, {input_name: input_data})
    reshape_dims = dict(zip(reshape_names, [output.shape for output in outputs]))

    to_replace_initializers = []

    # 遍历图中的所有节点
    for node in model.graph.node:
        # 如果节点是Reshape操作
        if node.op_type == "Reshape":
            # 获取形状的输入名称
            shape_input_name = node.input[1]
            # 在图中找到这个输入
            for initializer in model.graph.initializer:
                if initializer.name == shape_input_name:
                    # 获取形状的值
                    shape = onnx.numpy_helper.to_array(initializer)
                    # 计算新的形状
                    new_shape = np.array(
                        reshape_dims[node.output[0]], dtype=shape.dtype
                    )
                    new_shape[0] = -1
                    # 更新形状的值
                    new_tensor = onnx.numpy_helper.from_array(
                        new_shape, name=f"{node.name}/shape"
                    )
                    # 替换原始的初始化器
                    to_replace_initializers.append((initializer, new_tensor))
                    node.input[1] = new_tensor.name
    for initializer, new_tensor in to_replace_initializers:
        if initializer in model.graph.initializer:
            model.graph.initializer.remove(initializer)
        model.graph.initializer.append(new_tensor)
    return model


def replace_squeeze_and_unsqueeze(model: onnx.ModelProto) -> onnx.ModelProto:
    """用Reshape算子替换onnx中的squeeze和unsqueeze算子

    Args:
        model (onnx.ModelProto): 需要重新编排Reshape节点的ONNX模型

    Returns:
        onnx.ModelProto: 重新编排后的ONNX模型
    """
    # 复制一份模型，防止修改原模型
    model = onnx.load_model_from_string(model.SerializeToString())

    # 运行模型，获取每个Reshape节点的输出维度
    temp_model = onnx.load_model_from_string(model.SerializeToString())
    node_names = []
    for node in temp_model.graph.node:
        if node.op_type in ["Squeeze", "Unsqueeze"]:
            temp_model.graph.output.extend(
                [
                    onnx.helper.make_tensor_value_info(
                        node.output[0], onnx.TensorProto.FLOAT, [None]
                    )
                ]
            )
            node_names.append(node.output[0])
    input_name = temp_model.graph.input[0].name
    input_shape = [
        dim.dim_value for dim in temp_model.graph.input[0].type.tensor_type.shape.dim
    ]
    input_data = np.random.random_sample(input_shape).astype(np.float32)
    sess = onnxruntime.InferenceSession(
        temp_model.SerializeToString(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    outputs = sess.run(node_names, {input_name: input_data})
    reshape_dims = dict(zip(node_names, [output.shape for output in outputs]))

    new_nodes = []
    # 遍历原始模型的所有节点
    for node in model.graph.node:
        # 如果节点是Squeeze或Unsqueeze操作
        if node.op_type in ["Squeeze", "Unsqueeze"]:
            new_shape = reshape_dims[node.output[0]]
            reshape_param = onnx.helper.make_tensor(
                name=f"{node.name}_reshape_param",
                data_type=onnx.TensorProto.INT64,
                dims=[len(new_shape)],
                vals=new_shape,
            )
            # 创建一个新的Reshape操作
            reshape_node = onnx.helper.make_node(
                "Reshape",
                inputs=[node.input[0], reshape_param.name],
                outputs=node.output,
                name=node.name,
            )
            # 将新的Reshape操作和shape参数添加到图中
            model.graph.initializer.append(reshape_param)

            # 将新的Reshape节点添加到新的模型图中
            new_nodes.append(reshape_node)
        else:
            # 如果节点不是Squeeze或Unsqueeze操作，直接添加到新的模型图中
            new_nodes.append(node)

    # 创建一个新的图
    new_graph = onnx.helper.make_graph(
        new_nodes,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
    )

    # 替换原始的图
    model.graph.CopyFrom(new_graph)

    return model


def merge_slice(model: onnx.ModelProto) -> onnx.ModelProto:
    """将onnx模型中的Slice节点合并为Split节点

    Args:
        model (onnx.ModelProto): 需要进行操作的onnx模型

    Returns:
        onnx.ModelProto: 合并后的onnx模型
    """
    model = onnx.load_model_from_string(model.SerializeToString())

    # 获取模型中的initializer
    initializers = dict((init.name, init) for init in model.graph.initializer)

    # 创建一个列表来存储每个Slice节点的名称
    slice_dict = {}
    for node in model.graph.node:
        if node.op_type == "Slice":
            if node.input[0] not in slice_dict:
                slice_dict[node.input[0]] = []
            starts_node = initializers[node.input[1]]
            starts = onnx.numpy_helper.to_array(starts_node)[0]
            ends_node = initializers[node.input[2]]
            ends = onnx.numpy_helper.to_array(ends_node)[0]
            axis_node = initializers[node.input[3]]
            axis = onnx.numpy_helper.to_array(axis_node)[0]
            slice_dict[node.input[0]].append(
                dict(node=node, starts=starts, ends=ends, axis=axis)
            )
        elif node.op_type == "Split":
            pass

    # 获取所有的Slice节点的父节点的输出维度
    temp_model = onnx.load_model_from_string(model.SerializeToString())
    output_names = list(slice_dict.keys())
    for output_name in output_names:
        temp_model.graph.output.extend(
            [
                onnx.helper.make_tensor_value_info(
                    output_name, onnx.TensorProto.FLOAT, [None]
                )
            ]
        )
    input_name = temp_model.graph.input[0].name
    input_shape = [
        dim.dim_value for dim in temp_model.graph.input[0].type.tensor_type.shape.dim
    ]
    input_data = np.random.random_sample(input_shape).astype(np.float32)
    sess = onnxruntime.InferenceSession(
        temp_model.SerializeToString(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    outputs = sess.run(output_names, {input_name: input_data})
    output_dims = dict(zip(output_names, [output.shape for output in outputs]))

    # 判断哪些slice操作可以合并
    to_merge = {}
    for input_name, slice_nodes in slice_dict.items():
        # 按照slice的起始位置对slice_nodes进行排序
        slice_nodes.sort(key=lambda node: node["starts"])
        # 判断是否有多个slice操作
        if len(slice_nodes) == 1:
            continue
        # 判断slice的axis是否相同
        for i in range(1, len(slice_nodes)):
            if slice_nodes[i]["axis"] != slice_nodes[0]["axis"]:
                continue
        axis = slice_nodes[0]["axis"]
        # 判断slice的start和end是否连续
        for i in range(1, len(slice_nodes)):
            if slice_nodes[i]["starts"] != slice_nodes[i - 1]["ends"]:
                continue
        # 判断是否使用slices的输出合并是否是父节点的输出
        total_dim = 0
        for i in range(len(slice_nodes)):
            total_dim += slice_nodes[i]["ends"] - slice_nodes[i]["starts"]
        if output_dims[input_name][axis] != total_dim:
            continue
        for i in range(len(slice_nodes)):
            to_merge[slice_nodes[i]["node"].name] = slice_nodes

    # 创建一个集合来存储已经处理过的Slice节点的名称
    processed = set()

    new_nodes = []
    # 遍历图中的所有节点
    for node in model.graph.node:
        if node.name in processed:
            continue
        if node.op_type != "Slice" or node.name not in to_merge:
            new_nodes.append(node)
            continue
        brother = to_merge[node.name]
        axis = brother[0]["axis"]
        splits = []
        for i in range(len(brother)):
            splits.append(brother[i]["ends"] - brother[i]["starts"])
        # 获取onnx的版本
        onnx_opset_version = model.opset_import[0].version
        if onnx_opset_version < 13:
            new_node = onnx.helper.make_node(
                "Split",
                name=f"{node.name}/split",
                inputs=[node.input[0]],
                outputs=[parent_node["node"].output[0] for parent_node in slice_nodes],
                axis=axis,
                split=splits,
            )
        else:
            split_param = onnx.helper.make_tensor(
                name=f"{node.name}/split_param",
                data_type=onnx.TensorProto.INT64,
                dims=[len(splits)],
                vals=splits,
            )
            model.graph.initializer.append(split_param)
            new_node = onnx.helper.make_node(
                "Split",
                name=f"{node.name}/split",
                inputs=[node.input[0], split_param.name],
                outputs=[parent_node["node"].output[0] for parent_node in slice_nodes],
                axis=axis,
            )
        new_nodes.append(new_node)
        processed.update(parent_node["node"].name for parent_node in slice_nodes)

    # 创建一个新的图
    new_graph = onnx.helper.make_graph(
        new_nodes,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
    )

    # 替换原始的图
    model.graph.CopyFrom(new_graph)

    return model


def reshape_output(model: onnx.ModelProto) -> onnx.ModelProto:
    """修改onnx模型的输出尺寸，使得输出是4维的，在batch后填充1

    Args:
        model (onnx.ModelProto): 需要进行操作的onnx模型

    Returns:
        onnx.ModelProto: 修改后的onnx模型
    """
    model = onnx.load_model_from_string(model.SerializeToString())

    # 使用onnxruntime来运行模型，并获取每个输出的维度
    output_names = [output.name for output in model.graph.output]
    input_name = model.graph.input[0].name
    input_shape = [
        dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
    ]
    input_data = np.random.random_sample(input_shape).astype(np.float32)
    sess = onnxruntime.InferenceSession(
        model.SerializeToString(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    outputs = sess.run(output_names, {input_name: input_data})
    output_dims = dict(zip(output_names, [output.shape for output in outputs]))

    output_node_map = {}
    for node in model.graph.node:
        for output in node.output:
            output_node_map[output] = node

    # 遍历模型的所有输出
    for output in model.graph.output:
        # 获取输出的维度
        dims = output_dims[output.name]
        # 如果维度不足4
        if len(dims) < 4:
            previous_node: onnx.NodeProto = output_node_map[output.name]
            previous_node_output_name = f"{previous_node.name}/output"
            previous_node.output.remove(output.name)
            previous_node.output.append(previous_node_output_name)

            new_shape = [-1] + [1] * (4 - len(dims)) + list(dims)[1:]
            reshape_param = onnx.helper.make_tensor(
                name=f"{output.name}_reshape_param",
                data_type=onnx.TensorProto.INT64,
                dims=[len(new_shape)],
                vals=new_shape,
            )
            # 创建一个新的Reshape操作
            reshape_node = onnx.helper.make_node(
                "Reshape",
                inputs=[previous_node_output_name, reshape_param.name],
                outputs=[output.name],
                name=f"{output.name}_reshape",
            )
            # 将新的Reshape操作和shape参数添加到图中
            model.graph.node.append(reshape_node)
            model.graph.initializer.append(reshape_param)
            for _ in range(len(dims)):
                output.type.tensor_type.shape.dim.pop()
            for dim in [dims[0], *new_shape[1:]]:
                output.type.tensor_type.shape.dim.append(
                    onnx.TensorShapeProto.Dimension(dim_value=dim)
                )

    return model


def process(args):
    onnx_model = onnx.load(args.onnx_file)
    for fn in args.modifiers:
        st.write(f"Applying {fn.__name__}")
        onnx_model: onnx.ModelProto = fn(onnx_model)
        if args.save_intermediate:
            st.download_button(
                f"Download result of {fn.__name__}",
                data=onnx_model.SerializeToString(),
                file_name=f"{args.onnx_file.name}_{fn.__name__}.onnx",
            )
    st.write("Checking model")
    onnx.checker.check_model(onnx_model)
    st.write("Finished")
    st.download_button(
        "Download result",
        data=onnx_model.SerializeToString(),
        file_name=f"{args.onnx_file.name}_result.onnx",
    )


def main():
    st.title("ONNX Modifier")
    onnx_file = st.file_uploader("Choose an ONNX file", type="onnx")
    if "modifiers" not in st.session_state:
        st.session_state.modifiers = [
            simplify,
            modify_reshape,
            replace_squeeze_and_unsqueeze,
            merge_slice,
            reshape_output,
            simplify,
        ]
    for modifier in [
        "simplify",
        "modify_reshape",
        "replace_squeeze_and_unsqueeze",
        "merge_slice",
        "reshape_output",
    ]:
        if st.button(f"Add {modifier}"):
            st.session_state.modifiers.append(eval(modifier))
    st.write("Steps:", list(map(lambda x: x.__name__, st.session_state.modifiers)))
    if st.button("Clear modifiers"):
        st.session_state.modifiers = []
    save_intermediate = st.checkbox("Save intermediate results")
    if st.button("Start"):
        args = EasyDict(
            onnx_file=onnx_file,
            modifiers=st.session_state.modifiers,
            save_intermediate=save_intermediate,
        )
        process(args)


main()
