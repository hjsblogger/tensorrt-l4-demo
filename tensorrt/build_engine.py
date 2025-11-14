import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_file_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    # Use new API only
    config = builder.create_builder_config()

    # Set workspace memory
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Enable FP16 on L4
    config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 10), (32, 10), (64, 10))
    config.add_optimization_profile(profile)

    # Build engine
    engine = builder.build_engine(network, config)
    return engine

engine = build_engine("simple_model.onnx")
print("TensorRT engine built:", engine is not None)
