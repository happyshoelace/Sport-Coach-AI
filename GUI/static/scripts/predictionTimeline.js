const classes = ["En Garde", "Fleche", "Lunge", "Step"]; // Match backend order!
const timelineData = window.all_predictions || [];

const ctx = document.getElementById("timelineChart").getContext("2d");

// Get the predicted class index for each frame
const predictedClassIndices = timelineData.map(arr => arr.indexOf(Math.max(...arr)));

// Prepare data for scatter plot
const scatterData = predictedClassIndices.map((classIdx, frameIdx) => ({
  x: frameIdx + 1,
  y: classIdx
}));

const timelineChart = new Chart(ctx, {
  type: "scatter",
  data: {
    datasets: [{
      label: "Predicted Class Over Time",
      data: scatterData,
      backgroundColor: "rgba(54, 162, 235, 0.7)",
      pointRadius: 6,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `Frame ${ctx.parsed.x}: ${classes[ctx.parsed.y]}`,
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Frame" },
        min: 1,
        max: timelineData.length,
        ticks: { stepSize: 1, precision: 0 }
      },
      y: {
        type: "linear",
        title: { display: true, text: "Class" },
        min: -0.5,
        max: classes.length - 0.5,
        ticks: {
          stepSize: 1,
          callback: (val) => classes[val] || "",
        },
      },
    },
  },
});
