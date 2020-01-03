import Base.getindex
import Base.setindex!
struct DiskFourTensor
	"""
	Data structure for Rank 4 tensors
	"""
	fname::String
	dname::String
	sz1::Int
	sz2::Int
	sz3::Int
	sz4::Int
end

function DiskFourTensor(fname::String,dtype::Type,sz1::Int,sz2::Int,sz3::Int,sz4::Int,mode::String="r+")
	"""
	Constructor for DiskFourTensor objects
	"""
	file = h5open(fname,mode)
	if (mode == "w") | (mode == "w+")
		dataset = d_create(file,"data",datatype(dtype),dataspace(sz1))
	else
		dataset = file["data"]
	end
	DiskFourTensor(fname,"data",sz1,sz2,sz3,sz4)
end
# >>> overload getindex ( A[i,j,k,l] ) syntax
function ranger(inp::Union{UnitRange{Int64},Int64})
	if typeof(inp) == Int64
		return UnitRange(inp:inp)
	else
		return inp
	end
end
function getindex(dtens::DiskFourTensor,i1::UnitRange{Int64},i2::UnitRange{Int64},
				  i3::UnitRange{Int64},i4::UnitRange{Int64})
	h5open(dtens.fname,"r") do fid
		fid["$dtens.dname"][i1,i2,i3,i4]
	end
end
function getindex(dtens::DiskFourTensor,
				  i1::Union{UnitRange{Int64},Int64},
				  i2::Union{UnitRange{Int64},Int64},
				  i3::Union{UnitRange{Int64},Int64},
				  i4::Union{UnitRange{Int64},Int64})
	h5open(dtens.fname,"r") do fid
		fid["$dtens.dname"][ranger(i1),ranger(i2),ranger(i3),ranger(i4)]
	end
end
# <<< 

# >>> overload setindex! ( A[i,j,k,l] = 2.0 )
function setindex!(dtens::DiskFourTensor,val,
				   i1::UnitRange{Int64},i2::UnitRange{Int64},
				   i3::UnitRange{Int64},i4::UnitRange{Int64})
	h5open(dtens.fname,"r+") do fid
		fid["$dtens.dname"][i1,i2,i3,i4] = val
	end
end
function setindex!(dtens::DiskFourTensor,val,
				  i1::Union{UnitRange{Int64},Int64},
				  i2::Union{UnitRange{Int64},Int64},
				  i3::Union{UnitRange{Int64},Int64},
				  i4::Union{UnitRange{Int64},Int64})
	h5open(dtens.fname,"r+") do fid
		fid["$dtens.dname"][ranger(i1),ranger(i2),ranger(i3),ranger(i4)] = val
	end
end

function blockfill!(dtens::DiskFourTensor,val)
	"""
	Fill a DiskFourTensor with a single value.
	"""
	A = zeros(Float64,dtens.sz1,dtens.sz2,
			  dtens.sz3,dtens.sz4)
	A .= val
	h5write(dtens.fname,"$dtens.dname",A)
end

function tensordot(dtens1::DiskFourTensor,dtens2::DiskFourTensor)
end
