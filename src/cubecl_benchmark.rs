use std::marker::PhantomData;

use cubecl::benchmark::{Benchmark, };
use cubecl::{future, prelude::*};
use crate::cubecl_gputensor;
use cubecl_gputensor::GpuTensor; // Change to the path of your own module containing the GpuTensor

pub struct ReductionBench<R: Runtime, F: Float + CubeElement> {
    pub input_shape: Vec<usize>,
    pub client: ComputeClient<R::Server, R::Channel>,
    pub _f: PhantomData<F>,
}

const LINE_SIZE: u32 = 4;
impl<R: Runtime, F: Float + CubeElement> Benchmark for ReductionBench<R, F> {
    type Input = GpuTensor<R, F>;
    type Output = GpuTensor<R, F>;

    fn prepare(&self) -> Self::Input {
        GpuTensor::<R, F>::arange(self.input_shape.clone(), &self.client)
    }

    fn name(&self) -> String {
        format!("{}-reduction-{:?}", R::name(&self.client), self.input_shape).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn execute(&self, input: Self::Input) -> Result<GpuTensor<R, F>, String> {
        let output_shape: Vec<usize> = vec![self.input_shape[0]];
        
        let output = GpuTensor::<R, F>::empty(output_shape, &self.client);
        
        unsafe {
            reduce_matrix::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(self.input_shape[0] as u32, 1, 1), // Add parallelization on the first dimension
                input.into_tensor_arg(1),
                output.into_tensor_arg(1),
            );
        }

        Ok(output)
    }
}

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut acc = Line::new(F::new(0.0f32));
    for i in 0..input.shape(1) / LINE_SIZE {
        acc += input[UNIT_POS_X * input.stride(0) + i];
    }
    output[UNIT_POS_X] = acc;
    
}