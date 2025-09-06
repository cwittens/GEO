function get_grid_periodic(N, min, max)
    full_range = range(min, max, length=N + 1)
    return range(full_range[1], stop=full_range[end-1],
        length=N)
end

function get_grid(N, min, max)
    return  range(min, max, length=N) 
end