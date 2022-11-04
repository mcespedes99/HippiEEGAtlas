%% Entire brain
% figure
% p = patch('Faces',FacesRight,'Vertices', ...
%           NodesRight,'FaceVertexCData',NodesRegionRight);
% p.FaceColor = 'flat';
% p.LineStyle = 'none';
% p2 = patch('Faces',FacesLeft,'Vertices', ...
%           NodesLeft,'FaceVertexCData',NodesRegionLeft);
% p2.FaceColor = 'flat';
% p2.LineStyle = 'none';
% set(gca,'XColor',[1 1 1]);
% set(gca,'yColor',[1 1 1]);
% axis equal tight;
% colormap 'jet';
%% Subregion
figure
NodesSubregionRight = areaplot(35, NodesRegionRight);
% NodesSubregionRight = areasplot([26,27], NodesRegionRight);

p3 = patch('Faces',FacesRight,'Vertices', ...
          NodesRight,'FaceVertexCData',NodesSubregionRight);
p3.FaceColor = 'flat';
p3.LineStyle = '--';
p3.FaceAlpha = 0.3;
p3.EdgeAlpha = 0.3;
rotate(p3, [0 1 0], 90);
rotate(p3, [0 0 1], 90);

% NodesSubregionLeft = areaplot(4, NodesRegionLeft);
% p4 = patch('Faces',FacesLeft,'Vertices', ...
%           NodesLeft,'FaceVertexCData',NodesSubregionLeft);
% p4.FaceColor = 'flat';
% p4.LineStyle = 'none';
set(gca,'XColor',[1 1 1]);
set(gca,'yColor',[1 1 1]);
axis equal tight;


function area_to_plot = areaplot(NumRegionPlot, NodesRegion)
    area_to_plot = zeros(size(NodesRegion));
    ids = NodesRegion == NumRegionPlot;
    area_to_plot(ids) = 10;
end
function areas_to_plot = areasplot(NumRegionsPlot, NodesRegion)
    areas_to_plot = zeros(size(NodesRegion));
    ids = NodesRegion == NumRegionsPlot(1);
    areas_to_plot(ids) = 10;
    ids2 = NodesRegion == NumRegionsPlot(2);
    areas_to_plot(ids2) = 20;
end
