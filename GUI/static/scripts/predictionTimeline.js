const classes = ["En Garde", "Fleche", "Lunge", "Step"];

const timelineContext = document.getElementById("timelineChart");

const datasets = classes.map((cls, laneIndex) => ({
  label: cls,
  data: [],
  stepped: true,
  borderColor: `hsl(${laneIndex * 90}, 70%, 50%)`,
  backgroundColor: `hsl(${laneIndex * 90}, 70%, 50%)`,
  fill: false,
  showLine: false,
  spanGaps: true,
  tension: 0,
  pointRadius: 10,
  borderWidth: 3,
}));

const timelineChart = new Chart(timelineContext, {
  type: "line",
  data: { datasets },
  showLine: false,
  options: {
    responsive: true,
    interaction: {
      mode: "nearest",
      intersect: false,
    },
    plugins: {
      legend: { position: "right" },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.dataset.label} at frame = ${ctx.parsed.x}`,
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Frames" },
        ticks: { stepSize: 1, precision: 0 },
        offset: true,
        min: 0,
        max: 10,
      },
      y: {
        type: "linear",
        title: { display: true, text: "Class" },
        ticks: {
          stepSize: 1,
          callback: (val) => classes[val - 1] || "",
          min: 0.5,
          max: classes.length + 0.5,
        },
      },
    },
  },
});

function updateTimeline(predictions) {
  const n = predictions.length;

  classes.forEach((cls, laneIndex) => {
    const result = predictions.map((pred, time) => {
      return pred[laneIndex] === 1
        ? { x: time, y: laneIndex + 1 }
        : { x: time, y: null }; // keep the nulls here to break lines
    });
    console.log(result);
    timelineChart.data.datasets[laneIndex].data = result;
  });

  timelineChart.options.scales.x.max = n > 0 ? n - 1 : 10;
  timelineChart.update();
}

updateTimeline([
  [0, 0, 1, 0],
  [0, 0, 1, 0],
  [0, 0, 1, 0],
  [0, 0, 1, 0],
  [0, 0, 1, 0],
  [0, 0, 1, 0],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
]);

console.log("updated timeline!");
