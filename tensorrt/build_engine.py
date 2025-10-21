import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_file_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None
        
        builder.max_workspace_size = 1 << 30
        builder.fp16_mode = True  # For 40% reduction on L4
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 10), (32, 10), (64, 10))
        config.add_optimization_profile(profile)
        
        engine = builder.build_engine(network, config)
        return engine

engine = build_engine("simple_model.onnx")
print("TensorRT engine built:", engine is not None)