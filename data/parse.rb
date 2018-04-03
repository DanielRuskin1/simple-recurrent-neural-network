def make_regex(list)
  list.map { |p| "\\#{p}" }.join
end

def split_to_array(str, list)
  str.split(/([#{make_regex(list)}])/)
end

# Read in raw text
raw_text = File.read("raw.txt")

# Make it all non-cap
raw_text = raw_text.downcase

# Remove quotes
raw_text = raw_text.gsub("“", "").gsub("”", "")

# Split by newline to get distinct sentences
raw_text = raw_text.split("\n")

# Split by punctuation, but add it back in.
parsed_text = []
raw_text.each do |raw_text_entry|
  sentence_split_punct = [".", "?", "!"]
  split_to_array(raw_text_entry, sentence_split_punct).each_slice(2) do |sentence|
    parsed = sentence.join.strip
    other_punct = [".", ",", "?", "!", "'", ":", ";", "-"]
    end_punct =  [".", "?", "!"]
    parsed_text << split_to_array(parsed, other_punct).join(" ").squeeze(" ").gsub(/[#{make_regex(end_punct)}]/, "END_TOKEN")
  end
end

# Remove sentences less than or equal to 5 words (except for END_TOKEN)
parsed_text = parsed_text.delete_if do |d|
  d.gsub("END_TOKEN", "").split(" ").size <= 5
end

# Save to new newline-split file of sentences
new_file = File.open("parsed.txt", "w+")
parsed_text.each { |parsed| new_file.puts(parsed) }
new_file.close
