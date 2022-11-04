regionFound = closestRegion(RegionName, NodesRegionLeft, NodesLeft, -5, -96.5, 20);

%Function to find the closest node to an specific coordinate: Necessary to plot
function region = closestRegion(RegionName,NodesRegion,Nodes,x,y,z)
    [~,id] = min((Nodes(:,1) - x).^2 + (Nodes(:,2) - y).^2 + (Nodes(:,3) - z).^2);
    regionNum = NodesRegion(id);
    region = RegionName(regionNum);
end
