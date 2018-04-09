# Generate 1k increasing sequences
num_exs = 1000
start = 0
end_val = 15
data = []
(0..num_exs - 1).each do |ex_num|
  start_here = rand(start..(end_val - 2))
  end_here = rand((start_here+1)..end_val)
  data << (start_here..end_here).to_a
end

# Save to file
require 'csv'
File.open("data.csv", "w") do |file|
  data.each { |d| file << d.join(" ") << "\n" }
end
