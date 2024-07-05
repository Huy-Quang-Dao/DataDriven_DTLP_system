using LinearAlgebra
using Plots
using LaTeXStrings

function read_vector_from_file(filename)
    data = [];
    open(filename, "r") do file
        for line in eachline(file)
            push!(data, parse(Float64, strip(line)));
        end
    end
    return data
end

filename = "cmake-build-debug/output.txt";
vector_data = read_vector_from_file(filename);
n_learn = length(vector_data);

# t=1:1:n_learn;
# plot(t[:,1],vector_data, label=L"output",xlabel = "Iteration", linewidth=1)

t=collect(1:1:n_learn);
anim1 = @animate for i = 1:n_learn
    plot(t[1:i], vector_data[1:i], label=L"\|K - K^* \| ",xlabel = "Iteration",ylabel = L"\|K - K^* \| ", linewidth=2,legendfontsize=10,line=:solid, marker=:circle, color=:magenta,lw =2,markersize=7)
end
gif(anim1, "Solution.gif", fps = 3)

