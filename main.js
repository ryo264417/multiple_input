function loadImage(url) {
  return new Promise(function(resolve, reject) {
    const img = new Image()
    img.src = url
    img.addEventListener('load', () => resolve(img))
  })
}
const modelPromise0 = tf.loadModel('./vgg16_model_3/model.json')
//const modelPromise1 = tf.loadLayersModel('./vgg16_model5/model7/model.json')
/*
const modelPromise2 = tf.loadModel('./vgg16_model4/2/model.json')
const modelPromise3 = tf.loadModel('./vgg16_model4/3/model.json')
const modelPromise4 = tf.loadModel('./vgg16_model4/4/model.json')
*/

document.addEventListener('DOMContentLoaded', async () => {
  const canvas1 = document.getElementById('canvas1')
  const ctx1 = canvas1.getContext('2d')
  const file1 = document.getElementById('file1')

  const canvas2 = document.getElementById('canvas2')
  const ctx2 = canvas2.getContext('2d')
  const file2 = document.getElementById('file2')

  const canvas3 = document.getElementById('canvas3')
  const ctx3 = canvas3.getContext('2d')
  const file3 = document.getElementById('file3')
  const output = document.getElementById('output')

  file1.addEventListener('change', async () => {
  const dataURL1 = URL.createObjectURL(file1.files[0])
  const img1 = await loadImage(dataURL1)
  ctx1.drawImage(img1, 0, 0, 150, 150)


  file2.addEventListener('change', async () => {
  const dataURL2 = URL.createObjectURL(file2.files[0])
  const img2 = await loadImage(dataURL2)
  ctx2.drawImage(img2, 0, 0, 150, 150)


  file3.addEventListener('change', async () => {
  const dataURL3 = URL.createObjectURL(file3.files[0])
  const img3 = await loadImage(dataURL3)
  ctx3.drawImage(img3, 0, 0, 150, 150)


    const model0 = await modelPromise0
  //  const model1 = await modelPromise1
/*
    const model2 = await modelPromise2
    const model3 = await modelPromise3
    const model4 = await modelPromise4
*/

    const prediction = tf.tidy(() => {
      let input1 = tf.fromPixels(ctx1.getImageData(0, 0, 150, 150)).resizeNearestNeighbor([224,224]);
      input1 = tf.cast(input1, 'float32').div(tf.scalar(255));
      input1 = input1.expandDims();

      let input2 = tf.fromPixels(ctx2.getImageData(0, 0, 150, 150)).resizeNearestNeighbor([224,224]);
      input2 = tf.cast(input2, 'float32').div(tf.scalar(255));
      input2 = input2.expandDims();

      let input3 = tf.fromPixels(ctx3.getImageData(0, 0, 150, 150)).resizeNearestNeighbor([224,224]);
      input3 = tf.cast(input3, 'float32').div(tf.scalar(255));
      input3 = input3.expandDims();
      return  model0.predict([input1, input2, input3]).dataSync()[0];
    })

    if (prediction  > 0.5) {
      output.innerHTML = `Ideal ${prediction * 100}%`
    } else {
      output.innerHTML = `Not ${100 - prediction * 100}%`
    }
    })
  })
})
})
