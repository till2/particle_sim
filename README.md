## 
# <div align="center">Agentenbasierte Simulation der COVID-19 Pandemie <br/> mit einem SEIR-Modell </div>
######  <div align="center">Ein Projekt im Modul Simulation & Modellierung von Till Zemann & Ben Kampmann.</div>


##### <div align="center">![showcase](https://user-images.githubusercontent.com/89709351/216829859-7cd902c5-0c30-4fed-9d70-903ff3b0aac7.gif) Our Covid-19 Particle Simulator.</div>


## How it works:

The infectious particles can infect healthy ("susceptible") particles. After an incubation period, they become infectious too. We use an SEIR model to simulate the pandemic:

- S = Susceptible
- E = Exposed (infected but not infectious)
- I = Infectious
- R = Recovered/Removed

![timeline](https://user-images.githubusercontent.com/89709351/216829246-ff6f2c29-fe20-4dc4-90d9-a2bfb8f7a3e6.png)

## How to run our code

1. Start the [demo.ipynb](demo.ipynb) Notebook
2. Set `install_dependencies` to `True` in the first code cell to install all necessary libraries. You can set it to `False` for all runs that you start after the first one.

```py
# install all dependencies
install_dependencies = True # set to False after the first execution
```

3. Set your experiment parameters, for example:

```py
config = {
    "run_name": "my_experiment_1",
    "save_plots": True,
    "n_people": 500,
    "infection_prob": 0.3,
    "avg_incubation_time":500,
    "avg_infectious_time": 1000,
    "max_timestep": 8000,
    "start_seed": 1,
    "n_runs": 3,
    "speedup_factor": 5,
    "debug_mode": False,
    "FPS": 60,
}
```

4. For the first time that you execute our code, you have to compute the heatmaps (because the 150MB file is too large for an upload to GitHub). To do that, just set `use_precomputed_heatmaps` to `False` (the calculation might take a few minutes). In all runs that you start afterwards, please set it to `True` so that you use the saved heatmap and don't have to calculate it again.

```py
pf = Pathfinder(sim, use_precomputed_heatmaps=False) # set to True after the first execution
```

## BibTeX Citation

In case our software is used for future projects, you can refer to the following citation.

```txt
@software{zemann_kampmann_particlesim,
  author = {Till Zemann and Ben Kampmann},
  title = {Agentenbasierte Simulation der COVID-19 Pandemie mit einem SEIR-Modell},
  howpublished = {\url{https://github.com/till2/particle_sim}},
  year = {2023},
}
```
