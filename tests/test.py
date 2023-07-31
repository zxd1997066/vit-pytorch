
import argparse
import os
import sys
import time

import torch


def create_model_input(args):
    # model
    if args.arch == "Token-to-Token-ViT":
        from vit_pytorch.t2t import T2TViT
        # 224 input
        input = torch.randn(args.batch_size, 3, 224, 224)
        model = T2TViT(
            dim = 512,
            image_size = 224,
            depth = 5,
            heads = 8,
            mlp_dim = 512,
            num_classes = 1000,
            t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
        )
    elif args.arch == "LeViT":
        from vit_pytorch.levit import LeViT
        # 224 input
        input = torch.randn(args.batch_size, 3, 224, 224)
        model = LeViT(
            image_size = 224,
            num_classes = 1000,
            stages = 3,             # number of stages
            dim = (256, 384, 512),  # dimensions at each stage
            depth = 4,              # transformer of depth 4 at each stage
            heads = (4, 6, 8),      # heads at each stage
            mlp_mult = 2,
            dropout = 0.1
        )
    elif args.arch == "DeepViT":
        from vit_pytorch.deepvit import DeepViT
        input = torch.randn(args.batch_size, 3, 256, 256)
        model = DeepViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    elif args.arch == "CaiT":
        from vit_pytorch.cait import CaiT
        input = torch.randn(args.batch_size, 3, 256, 256)
        model = CaiT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 12,             # depth of transformer for patch to patch attention only
            cls_depth = 2,          # depth of cross attention of CLS tokens to patch
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05    # randomly dropout 5% of the layers
        )
    else:
        from vit_pytorch import ViT
        input = torch.randn(args.batch_size, 3, 256, 256)
        model = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    return model.eval().to(args.device), input

def test(args):

    model, input = create_model_input(args)

    # NHWC
    if args.channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
            input = input.contiguous(memory_format=torch.channels_last)
            print("---- Use NHWC model and intput.")
        except:
            pass
    # ipex
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        print("Running with IPEX...")
        if args.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        else:
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
    # jit
    if args.jit:
        with torch.no_grad():
            try:
                model = torch.jit.trace(model, input, check_trace=False)
                print("---- Use trace model.")
            except:
                model = torch.jit.script(model)
                print("---- Use script model.")
            if args.ipex:
                model = torch.jit.freeze(model)

    # H2D
    h2d_time = 0.0
    if args.device == 'cuda':
        h2d_time = time.time()
        input = input.to(args.device)
        h2d_time = time.time() - h2d_time

    # compute
    total_sample = 0
    total_time = 0.0
    with torch.no_grad():
        if args.profile:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=int(args.num_iter/2),
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                for i in range(args.num_iter):
                    tic = time.time()
                    preds = model(input)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    p.step()
                    toc = time.time()
                    # caculate time
                    print("Iteration: {}, inference time: {} sec.".format(i, toc - tic + h2d_time), flush=True)
                    if i >= args.num_warmup:
                        total_time += (toc - tic + h2d_time)
                        total_sample += args.batch_size
        else:
            for i in range(args.num_iter):
                tic = time.time()
                preds = model(input)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                toc = time.time()
                # caculate time
                print("Iteration: {}, inference time: {} sec.".format(i, toc - tic + h2d_time), flush=True)
                if i >= args.num_warmup:
                    total_time += (toc - tic + h2d_time)
                    total_sample += args.batch_size

    print("\n", "-"*20, "Summary", "-"*20)
    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {:.3f} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

    assert preds.shape == (1, 1000), 'correct logits outputted'

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                args.arch + '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT',
                        help='model architecture (default: ViT)')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--ipex', action='store_true', default=False, help='enable ipex')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--device', default="cpu", type=str, help='device')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    args = parser.parse_args()
    print(args)

    # start test
    if args.precision == "bfloat16":
        print("---- Use AMP bfloat16")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            test(args)
    elif args.precision == "float16":
        print("---- Use AMP float16")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.half):
            test(args)
    else:
        test(args)

