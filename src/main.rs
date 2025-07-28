mod gelu;
mod cubecl_gputensor;
mod cubecl_benchmark;
mod cubecl_example;
use std::marker::PhantomData;
use cubecl_example::CpuTensor;
use cubecl_gputensor::GpuTensor;
use gelu::launch as gelu_launch;
use cubecl_benchmark::{ReductionBench};
use cubecl::{benchmark::{Benchmark, TimingMethod}, prelude::*};


pub fn launch_bench<R:Runtime, F: Float + CubeElement>(device: &R::Device) {
    let client = R::client(&device);

    let bench1 = ReductionBench::<R,F> {
        input_shape: vec![64,256, 1024],
        client: client.clone(),
        _f: PhantomData
    };
    let bench2 = ReductionBench::<R,F> {
        input_shape: vec![64, 64, 4096],
        client: client.clone(),
        _f: PhantomData
    };

    for bench in [bench1, bench2] {
        println!("{:?}", bench.name());
        println!("{:?}", bench.run(TimingMethod::System));
    }
}

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    for i in 0..input.shape(0) {
        let mut acc = F::new(0.0f32);
        for j in 0..input.shape(1) {
            acc += input[i * input.stride(0) + j];
        }
        output[i * output.stride(0)] = acc;
    }
}

pub fn launch_gpu<R: Runtime, F: Float + CubeElement>(device: &R::Device) {
    let client = R::client(device);
    
    let input = GpuTensor::<R, F>::arange(vec![4,3], &client);
    let output = GpuTensor::<R, F>::arange(vec![4,3], &client);

    unsafe {
        reduce_matrix::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            input.into_tensor_arg(1),
            input.into_tensor_arg(1),
        )
    };
    println!(
        "Executed reduction with runtime {:?} => {:?}",
        R::name(&client),
        output.read(&client)
    );
}

fn launch_on_cpu() {
    fn reduce_matrix(input: &CpuTensor, output: &mut CpuTensor) {
        for i in 0..input.shape[0] {
            let mut acc = 0.0f32;
            for j in 0..input.shape[1] {
                acc += input.data[i * input.strides[0] + j];
            }
            output.data[i] = acc;
        }
        
    }
    let input_shape = vec![4,3];
    let output_shape = vec![4];
    let input = CpuTensor::arange(input_shape);
    let mut output = CpuTensor::empty(output_shape);

    reduce_matrix(&input, &mut output);
    println!("Input Tensor: {:?}", input.read());
    println!("Output Tensor: {:?}", output.read());
}
fn main() {
    // launch_on_cpu();
    // launch_gpu::<cubecl::wgpu::WgpuRuntime, f32>(&Default::default());
    // launch_bench::<cubecl::wgpu::WgpuRuntime, f32>(&Default::default());
    gelu_launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
    
}
