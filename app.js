const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let width, height, target;
let historyData = [];
let isAnimating = false;
let autoPlay = false;
let lastTime = 0;
let playbackSpeed = 0;

function resize() {
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = width;
  canvas.height = height;
  target = { x: width / 2, y: height / 2 };
}
window.addEventListener("resize", resize);
resize();

const btnStart = document.getElementById("startBtn");
const timelineContainer = document.getElementById("timeline-container");
const timeline = document.getElementById("timeline");
const genLabel = document.getElementById("genLabel");
const fitnessLabel = document.getElementById("fitnessLabel");

class Individual {
  constructor(x, y) {
    this.x = x !== undefined ? x : Math.random() * width;
    this.y = y !== undefined ? y : Math.random() * height;
    this.fitness = 0;
  }

  calcFitness() {
    const dx = this.x - target.x;
    const dy = this.y - target.y;
    this.fitness = Math.sqrt(dx * dx + dy * dy);
  }
}

function runGA(popSize, maxGen, patience) {
  let population = [];
  let localHistory = [];

  for (let i = 0; i < popSize; i++) {
    population.push(new Individual());
  }

  let bestOverallFitness = Infinity;
  let patienceCounter = 0;

  for (let gen = 0; gen <= maxGen; gen++) {
    let bestInd = null;
    let currentBestFitness = Infinity;

    population.forEach((ind) => {
      ind.calcFitness();
      if (ind.fitness < currentBestFitness) {
        currentBestFitness = ind.fitness;
        bestInd = ind;
      }
    });

    localHistory.push({
      gen: gen,
      bestFitness: currentBestFitness,
      population: population.map((ind) => ({
        x: ind.x,
        y: ind.y,
        fitness: ind.fitness,
      })),
    });

    if (Math.abs(bestOverallFitness - currentBestFitness) < 1.0) {
      patienceCounter++;
    } else {
      patienceCounter = 0;
      if (currentBestFitness < bestOverallFitness) {
        bestOverallFitness = currentBestFitness;
      }
    }

    if (patienceCounter >= patience) {
      console.log(`Convergência atingida na geração ${gen}`);
      break;
    }

    if (gen === maxGen) break;

    let newPopulation = [];
    newPopulation.push(new Individual(bestInd.x, bestInd.y));

    while (newPopulation.length < popSize) {
      let p1 = tournament(population);
      let p2 = tournament(population);

      let childX = (p1.x + p2.x) / 2;
      let childY = (p1.y + p2.y) / 2;

      const mutationRate = 0.1;
      if (Math.random() < mutationRate) {
        childX += (Math.random() - 0.5) * 100;
        childY += (Math.random() - 0.5) * 100;
      }

      newPopulation.push(new Individual(childX, childY));
    }
    population = newPopulation;
  }
  return localHistory;
}

function tournament(pop) {
  let idx1 = Math.floor(Math.random() * pop.length);
  let idx2 = Math.floor(Math.random() * pop.length);
  return pop[idx1].fitness < pop[idx2].fitness ? pop[idx1] : pop[idx2];
}

btnStart.addEventListener("click", () => {
  const popSize = parseInt(document.getElementById("popSize").value);
  const maxGen = parseInt(document.getElementById("maxGen").value);
  const patience = parseInt(document.getElementById("patience").value);
  const durationSec = parseFloat(document.getElementById("duration").value);

  historyData = runGA(popSize, maxGen, patience);

  timeline.max = historyData.length - 1;
  timeline.value = 0;
  timelineContainer.classList.remove("hidden");

  autoPlay = true;
  playbackSpeed = (historyData.length - 1) / (durationSec * 1000);
  lastTime = 0; 

  if (!isAnimating) {
    isAnimating = true;
    requestAnimationFrame(render);
  }
});


timeline.addEventListener("input", () => {
  autoPlay = false;
});

function render(timestamp) {
  if (!lastTime) lastTime = timestamp;
  const deltaTime = timestamp - lastTime;
  lastTime = timestamp;

  if (autoPlay) {
    let nextVal = parseFloat(timeline.value) + playbackSpeed * deltaTime;
    if (nextVal >= timeline.max) {
      nextVal = timeline.max;
      autoPlay = false;
    }
    timeline.value = nextVal;
  }

  ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
  ctx.fillRect(0, 0, width, height);

  ctx.beginPath();
  ctx.arc(target.x, target.y, 10, 0, Math.PI * 2);
  ctx.fillStyle = "#00ffcc";
  ctx.shadowBlur = 15;
  ctx.shadowColor = "#00ffcc";
  ctx.fill();
  ctx.shadowBlur = 0; 

  const timeVal = parseFloat(timeline.value);
  const currentGenIdx = Math.floor(timeVal);
  const nextGenIdx = Math.min(currentGenIdx + 1, historyData.length - 1);
  const progress = timeVal - currentGenIdx;

  const currentData = historyData[currentGenIdx];
  const nextData = historyData[nextGenIdx];

  genLabel.innerText = `Geração: ${currentData.gen}`;
  fitnessLabel.innerText = `Melhor Fitness: ${currentData.bestFitness.toFixed(2)}`;

  for (let i = 0; i < currentData.population.length; i++) {
    const pCurrent = currentData.population[i];
    const pNext = nextData.population[i] || pCurrent;

    const x = pCurrent.x + (pNext.x - pCurrent.x) * progress;
    const y = pCurrent.y + (pNext.y - pCurrent.y) * progress;

    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    const colorRatio = Math.max(0, 1 - pCurrent.fitness / (width / 2));
    ctx.fillStyle = `hsl(${colorRatio * 180 + 180}, 100%, 60%)`;
    ctx.fill();
  }

  requestAnimationFrame(render);
}
