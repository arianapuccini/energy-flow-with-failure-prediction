import Pkg
# first time add packages with Pkg.add("xxxxx")
using Printf, DifferentialEquations, Plots, LinearAlgebra, Graphs, GraphRecipes, CSV, DataFrames, Statistics

# compartment flow rates
# engine --> cooling:      0.30 (main cooling system)
# engine --> transmission: 0.10 (mechanical friction losses)
# cooling --> battery:     0.15 (waste heat reuse)
# battery leak:            0.08 (battery thermal losses)
# transmission leak:       0.10 (gearbox losses)

# column j = outflows from compartment j
# diagonal = -(sum of outflows), off-diagonal(i,j) = inflow to i from j
A = [-0.40  0.00  0.00  0.00;   # engine: loses 0.40 total
      0.30 -0.15  0.00  0.00;   # cooling: receives 0.30, loses 0.15 to battery
      0.00  0.15 -0.08  0.00;   # battery: receives 0.15, loses 0.08
      0.10  0.00  0.00 -0.10]   # transmission: receives 0.10, loses 0.10

# verify conservation: each column should sum to <= 0 (remaining goes to ambient)
col_sums = sum(A, dims=1)
println("Column sums (negative = energy leaving to ambient): ", col_sums)

function system!(du, u, p, t)
    du .= A * u
end

# all energy starts in engine
u0 = [100.0, 0.0, 0.0, 0.0]
tspan = (0.0, 30.0)

prob = ODEProblem(system!, u0, tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

# --- Plot 1: Subsystem dynamics ---
graph = plot(sol.t, sol[1,:], lw=2.5, label="Engine",       color=:firebrick)
plot!(sol.t, sol[2,:], lw=2.5, label="Cooling system",      color=:steelblue)
plot!(sol.t, sol[3,:], lw=2.5, label="Battery (recovered)", color=:seagreen)
plot!(sol.t, sol[4,:], lw=2.5, label="Transmission",        color=:darkorange)

# Show total energy in system (remainder has dissipated to ambient)
total_energy = sum(sol[i,:] for i in 1:4)
plot!(sol.t, total_energy, lw=1.5, label="Total (system)",
      color=:black, linestyle=:dash, alpha=0.6)

xlabel!("Time (s)")
ylabel!("Energy (normalized units)")
title!("Vehicle Subsystem Energy Dynamics")
display(graph)

# eigenvalues show time constants of the system
λ = eigvals(A)
println("\nSystem eigenvalues: ", λ)
println("Time constants (τ = -1/λ): ", -1 ./ real.(λ))
println("Dominant time constant: ", maximum(-1 ./ real.(λ[real.(λ) .< 0])), " s")

# linear compartmenal model
g = DiGraph(4)
add_edge!(g, 1, 2)   # Engine -> Cooling
add_edge!(g, 1, 4)   # Engine -> Transmission
add_edge!(g, 2, 3)   # Cooling -> Battery (waste heat recovery)

names = ["Engine", "Cooling", "Battery", "Transmission"]
weights = [0.30, 0.15, 0.10, 0.10]   # edge weights = flow rates

# summary
data = reduce(hcat, sol.u)
max_energy = maximum.(eachrow(data))
peak_time  = [sol.t[argmax(data[i,:])] for i in 1:4]
final_energy = data[:, end]
total_dissipated = 100.0 - sum(final_energy)

println("\nEnergy Summary")
for (i, name) in enumerate(names)
    @printf("%-15s  peak: %6.2f  at t=%.1fs  final: %5.2f\n",
            name, max_energy[i], peak_time[i], final_energy[i])
end
println("\nTotal energy dissipated to ambient: $(round(total_dissipated, digits=2))")
println("Energy recovery to battery: $(round(final_energy[3], digits=2)) ($(round(final_energy[3], digits=1))% of input)")

# sensor data from python
sensor_df = CSV.read("sensor_data.csv", DataFrame)

println("\n=== Sensor Data Summary ===")
println("Total samples: ", nrow(sensor_df))
println("Failure rate: ", round(mean(sensor_df.failure) * 100, digits=1), "%")

# map ODE compartment energy to sensor risk windows
# high engine energy in the ODE solution corresponds to high-temperature, high-vibration operating regimes in the sensor data
# this bins the ODE solution by energy level and compares failure rates

#engine compartment over time
engine_energy = sol[1, :]
max_e = maximum(engine_energy)

# normalize engine energy to a 0-1 risk proxy
energy_risk = engine_energy ./ max_e

# find time windows where engine energy > 50% of peak, representing high temperatures in the python data
high_energy_times = sol.t[energy_risk .> 0.5]
println("\nHigh engine energy period: t=0 to t=",
        round(maximum(high_energy_times), digits=1), "s")

# compare ODE risk window (>95 degrees C for engine temperature) to ML predicted failures
high_temp_failures = sensor_df[sensor_df.engine_temp .> 95, :]
failure_rate_high  = mean(high_temp_failures.failure)
failure_rate_all   = mean(sensor_df.failure)

println("\nFailure rate (all samples):        ",
        round(failure_rate_all * 100, digits=1), "%")
println("Failure rate (engine_temp > 95°C): ",
        round(failure_rate_high * 100, digits=1), "%")
println("Risk multiplier: ",
        round(failure_rate_high / failure_rate_all, digits=2), "x")

# from python
coef_df = CSV.read("model_coefficients.csv", DataFrame)
coefficients = Dict(zip(coef_df.feature, coef_df.coefficient))

println("\nLoaded model coefficients from Python:")
for (k, v) in coefficients
    println("  ", k, ": ", round(v, digits=4))
end

# Sigmoid function to represent probabilites
sigmoid(x) = 1 / (1 + exp(-x))

# make a single operating point using the python model
function fault_probability(engine_temp, vibration, oil_pressure, rpm)
    # Note: uses standardized inputs — you'd need scaler params for production
    # Here we use raw coefficients as a demonstration
    z = (coefficients["engine_temp"]  * engine_temp  +
         coefficients["vibration"]    * vibration    +
         coefficients["oil_pressure"] * oil_pressure +
         coefficients["rpm"]          * rpm)
    return sigmoid(z)
end

# evaluate fault risk at peak engine energy point
peak_idx = argmax(sol[1, :])
println("\n=== Fault Risk at Peak Engine Load ===")
println("Time: t=", round(sol.t[peak_idx], digits=1), "s")
println("Engine energy: ", round(sol[1, peak_idx], digits=1))

# use mean sensor values as baseline, elevated in the case of high energy
p_fault = fault_probability(105.0, 6.5, 36.0, 2200.0)
println("Estimated fault probability (high-load scenario): ",
        round(p_fault * 100, digits=1), "%")

# final plot
p1 = plot(sol.t, sol[1,:], lw=2, color=:firebrick, label="Engine energy")
p1 = vspan!(p1, [0, maximum(high_energy_times)],
            alpha=0.1, color=:red, label="High-risk window")
xlabel!("Time (s)")
ylabel!("Energy")
title!("ODE Energy + ML Risk Window")

p2 = histogram(sensor_df.failure_probability, bins=40,
               color=:steelblue, alpha=0.7, label="")
xlabel!("Predicted failure probability")
ylabel!("Count")
title!("ML Fault Probability Distribution")

combined = plot(p1, p2, layout=(1,2), size=(900, 400))
display(combined)