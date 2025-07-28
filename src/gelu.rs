use cubecl::prelude::*;

#[cube(launch_unchecked)]
/// A [Line] represents a contiguous series of elements where SIMD operations may be available.
fn gelu_array<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar(input[ABSOLUTE_POS]);
    }
}

#[cube]
fn gelu_scalar<F: Float>(x: Line<F>)  -> Line<F>  {
    let sqrt2 = F::new(comptime!(2.0f32.sqrt()));
    let tmp = x / Line::new(sqrt2);

    x * (Line::erf(tmp) + 1.0) / 2.0
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let vectorization = 4;
    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create(f32::as_bytes(input));

    unsafe {
        gelu_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1,1,1),
            CubeDim::new(input.len() as u32 / vectorization, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_handle, input.len(), vectorization as u8),
            ArrayArg::from_raw_parts::<f32>(&output_handle, input.len(), vectorization as u8),
        )
    };

    let bytes = client.read_one(output_handle.binding());
    let output = f32::from_bytes(&bytes);
    // GELU (Gaussian Error Linear Unit) activation function
    println!(
        "Executed GELU with runtime {:?} => {:?}",
        R::name(&client),
        output
    );

}